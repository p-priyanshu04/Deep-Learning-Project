import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedBLSTM(nn.Module):
    def __init__(self, embed_matrix, hidden_dim=50, dropout=0.2):
        super().__init__()
        vocab_size, embed_dim = embed_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding.weight = nn.Parameter(
            torch.tensor(embed_matrix, dtype=torch.float32), requires_grad=False
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim, hidden_size=hidden_dim, batch_first=True,
            bidirectional=True, dropout=dropout, num_layers=1,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids, lengths):
        embeds = self.dropout(self.embedding(token_ids))
        lengths_cpu = lengths.cpu().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            embeds, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        hidden_states, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=token_ids.size(1)
        )
        mask = (token_ids != 0).unsqueeze(-1).float()
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return hidden_states, pooled


class ClassicalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden_states, mask):
        u = torch.tanh(self.W(hidden_states))
        scores = self.v(u).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        alpha = F.softmax(scores, dim=1)
        attended = torch.bmm(alpha.unsqueeze(1), hidden_states).squeeze(1)
        return attended


class PositionalAttention(nn.Module):
    def __init__(self, lstm_hidden_dim, pos_hidden_dim):
        super().__init__()
        self.W_H = nn.Linear(lstm_hidden_dim, lstm_hidden_dim, bias=True)
        self.W_P = nn.Linear(pos_hidden_dim, lstm_hidden_dim, bias=False)
        self.v = nn.Linear(lstm_hidden_dim, 1, bias=False)

    def forward(self, a_hidden, p_vectors, a_mask):
        x = self.W_H(a_hidden) + self.W_P(p_vectors)
        scores = self.v(torch.tanh(x)).squeeze(-1)
        scores = scores.masked_fill(a_mask == 0, -1e9)
        alpha = F.softmax(scores, dim=1)
        attended = torch.bmm(alpha.unsqueeze(1), a_hidden).squeeze(1)
        return attended

# Helper function needed inside RNNPOA
def compute_position_influence(a_len, q_positions, max_a_len, sigma_scope=25, sigma_prime=0.1):
    d_hat = torch.zeros(max_a_len)
    if len(q_positions) == 0:
        return d_hat
    for p in range(a_len):
        influence = 0.0
        for qj in q_positions:
            if abs(p - qj) <= sigma_scope:
                influence += math.exp(-((p - qj) ** 2) / (2 * sigma_prime ** 2))
        d_hat[p] = influence
    if d_hat.max() > 0:
        d_hat = d_hat / d_hat.max()
    return d_hat

def batch_position_influence(a_lens, q_positions_list, max_a_len, sigma_scope=25, sigma_prime=0.1):
    batch_d = []
    for i in range(len(a_lens)):
        a_len_val = a_lens[i].item() if torch.is_tensor(a_lens[i]) else a_lens[i]
        d = compute_position_influence(a_len_val, q_positions_list[i], max_a_len, sigma_scope, sigma_prime)
        batch_d.append(d)
    return torch.stack(batch_d)

class RNNPOA(nn.Module):
    def __init__(self, embed_matrix, hidden_dim=50, pos_dim=50, sigma_scope=25, sigma_prime=0.1, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pos_dim = pos_dim
        self.sigma_scope = sigma_scope
        self.sigma_prime = sigma_prime
        
        self.encoder = SharedBLSTM(embed_matrix, hidden_dim, dropout)
        self.q_attn = ClassicalAttention(2 * hidden_dim)
        self.pos_attn = PositionalAttention(2 * hidden_dim, pos_dim)
        
        max_dist = 100 # MAX_A_LEN
        K = torch.zeros(pos_dim, max_dist)
        for u in range(max_dist):
            kernel_u = math.exp(-(u**2) / (2 * sigma_scope**2))
            K[:, u] = torch.normal(mean=kernel_u, std=sigma_prime, size=(pos_dim,))
        self.register_buffer('K', K)

    def _compute_p_vectors(self, a_len, q_pos, max_a_len):
        B = len(a_len)
        p_vectors = torch.zeros(B, max_a_len, self.pos_dim, device=self.K.device)
        for b in range(B):
            valid_len = a_len[b].item() if torch.is_tensor(a_len[b]) else a_len[b]
            for j in range(valid_len):
                for qp in q_pos[b]:
                    dist = abs(j - qp)
                    if dist < self.K.size(1):
                        p_vectors[b, j] += self.K[:, dist]
        return p_vectors

    def forward(self, q_ids, a_ids, q_len, a_len, q_pos):
        q_hidden, _ = self.encoder(q_ids, q_len)
        a_hidden, _ = self.encoder(a_ids, a_len)

        q_mask = (q_ids != 0).float()
        q_attended = self.q_attn(q_hidden, q_mask)

        p_vectors = self._compute_p_vectors(a_len, q_pos, a_ids.size(1))

        a_mask = (a_ids != 0).float()
        a_attended = self.pos_attn(a_hidden, p_vectors, a_mask)

        sim = torch.exp(-torch.sum(torch.abs(q_attended - a_attended), dim=1))
        return sim