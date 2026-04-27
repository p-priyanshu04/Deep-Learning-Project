import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedBLSTM(nn.Module):
    """
    Shared Bidirectional LSTM encoder.
    The same BLSTM weights encode both question and answer sentences,
    ensuring the representations live in the same space.

    Input:  (batch, seq_len)  token indices
    Output: (batch, seq_len, 2*hidden_dim)  hidden states
            (batch, 2*hidden_dim)  final pooled representation
    """
    def __init__(self, embed_matrix, hidden_dim=50, dropout=0.2):
        super().__init__()
        vocab_size, embed_dim = embed_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding.weight = nn.Parameter(
            torch.tensor(embed_matrix, dtype=torch.float32), requires_grad=False
        )  # Freeze pre-trained GloVe

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
            num_layers=1,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids, lengths):
        """
        Args:
            token_ids: (batch, max_len) padded token indices
            lengths:   (batch,) actual lengths
        Returns:
            hidden_states: (batch, max_len, 2*hidden_dim)
            pooled:        (batch, 2*hidden_dim) mean-pooled over valid positions
        """
        embeds = self.dropout(self.embedding(token_ids))  # (B, L, E)

        # Pack for efficient LSTM processing
        lengths_cpu = lengths.cpu().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            embeds, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        hidden_states, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=token_ids.size(1)
        )  # (B, L, 2H)

        # Mean-pool over valid (non-padded) positions
        mask = (token_ids != 0).unsqueeze(-1).float()  # (B, L, 1)
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return hidden_states, pooled


class RNNAVG(nn.Module):
    """
    Baseline: Shared BLSTM followed by mean pooling (no attention).
    """
    def __init__(self, embed_matrix, hidden_dim=50, dropout=0.2):
        super().__init__()
        self.encoder = SharedBLSTM(embed_matrix, hidden_dim, dropout)

    def forward(self, q_ids, a_ids, q_len, a_len, q_pos):
        _, q_pooled = self.encoder(q_ids, q_len)
        _, a_pooled = self.encoder(a_ids, a_len)
        
        # Manhattan distance similarity
        sim = torch.exp(-torch.sum(torch.abs(q_pooled - a_pooled), dim=1))
        return sim


class RNNATT(nn.Module):
    """
    Baseline: classical attention-based BLSTM (no positional influence).
    Used for comparison to show the benefit of positional attention.
    """
    def __init__(self, embed_matrix, hidden_dim=50, dropout=0.2):
        super().__init__()
        self.encoder = SharedBLSTM(embed_matrix, hidden_dim, dropout)
        self.W = nn.Linear(2*hidden_dim, 2*hidden_dim, bias=False)

    def forward(self, q_ids, a_ids, q_len, a_len, q_pos):
        _, q_pooled   = self.encoder(q_ids, q_len)
        a_hidden, _   = self.encoder(a_ids, a_len)

        Wq = self.W(q_pooled)
        scores = torch.bmm(a_hidden, Wq.unsqueeze(2)).squeeze(2)
        a_mask = (a_ids != 0).float()
        scores = scores.masked_fill(a_mask == 0, -1e9)
        alpha  = F.softmax(scores, dim=1)
        a_attended = torch.bmm(alpha.unsqueeze(1), a_hidden).squeeze(1)

        sim = torch.exp(-torch.sum(torch.abs(q_pooled - a_attended), dim=1))
        return sim


def compute_position_influence(a_len, q_positions, max_a_len, sigma_scope=25, sigma_prime=0.1):
    """
    Compute the position-aware influence vector d_hat for one sample.

    Math:
      d_hat[p] = sum_{q_j in Q_a} exp(-(p - q_j)^2 / (2 * sigma'^2))
      Only positions within |p - q_j| <= sigma_scope contribute.

    Args:
        a_len:        actual answer length
        q_positions:  list of answer-indices where question words appear
        max_a_len:    padded answer length
        sigma_scope:  propagation scope (paper: 15-35, default 25)
        sigma_prime:  Gaussian std dev (paper: 0.1)

    Returns:
        d_hat: (max_a_len,) tensor of influence values
    """
    d_hat = torch.zeros(max_a_len)
    if len(q_positions) == 0:
        return d_hat

    for p in range(a_len):
        influence = 0.0
        for qj in q_positions:
            if abs(p - qj) <= sigma_scope:
                influence += math.exp(-((p - qj) ** 2) / (2 * sigma_prime ** 2))
        d_hat[p] = influence

    # Normalise to [0, 1] to prevent scale issues
    if d_hat.max() > 0:
        d_hat = d_hat / d_hat.max()
    return d_hat


def batch_position_influence(a_lens, q_positions_list, max_a_len, sigma_scope=25, sigma_prime=0.1):
    """
    Compute position influence for a whole batch.
    Returns: (batch, max_a_len) tensor.
    """
    batch_d = []
    for i in range(len(a_lens)):
        d = compute_position_influence(
            a_lens[i].item() if torch.is_tensor(a_lens[i]) else a_lens[i],
            q_positions_list[i], max_a_len, sigma_scope, sigma_prime
        )
        batch_d.append(d)
    return torch.stack(batch_d)  # (B, L_a)


class PositionalAttention(nn.Module):
    """
    Positional Attention (RNN-POA) Layer.

    Combines classical bilinear attention with position-aware influence
    to weight the answer hidden states. The position-aware influence
    vector d_hat up-weights answer positions near question-word occurrences.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        # Bilinear scoring: s(h_a, h_q) = h_a^T W h_q
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, a_hidden, q_pooled, a_mask, d_hat):
        """
        Args:
            a_hidden: (B, L_a, 2H)  answer hidden states
            q_pooled: (B, 2H)       question representation
            a_mask:   (B, L_a)      1 for valid positions, 0 for padding
            d_hat:    (B, L_a)      position-aware influence vector

        Returns:
            attended: (B, 2H)  position-aware attended answer representation
        """
        # Bilinear attention scores: s(h_a_p, h_q) = h_a_p^T W h_q
        Wq = self.W(q_pooled)          # (B, 2H)
        scores = torch.bmm(
            a_hidden, Wq.unsqueeze(2)
        ).squeeze(2)                    # (B, L_a)

        # Mask padding positions with large negative value
        scores = scores.masked_fill(a_mask == 0, -1e9)

        # Classical attention weights
        exp_scores = torch.exp(scores - scores.max(dim=1, keepdim=True).values)  # numerical stability
        exp_scores = exp_scores * a_mask  # zero out padding

        # ── Positional modulation ──
        # Multiply by (1 + d_hat) so positions near question words get boosted
        modulated = exp_scores * (1.0 + d_hat)

        # Normalise to get positional attention weights
        alpha = modulated / modulated.sum(dim=1, keepdim=True).clamp(min=1e-9)  # (B, L_a)

        # Weighted sum of answer hidden states
        attended = torch.bmm(alpha.unsqueeze(1), a_hidden).squeeze(1)  # (B, 2H)
        return attended


class RNNPOA(nn.Module):
    """
    RNN with Positional Attention for Answer Selection.

    Architecture:
      1. Shared BLSTM encodes question and answer
      2. Position-aware influence propagation (Gaussian kernel)
      3. Positional attention over answer hidden states
      4. Manhattan distance similarity
    """
    def __init__(self, embed_matrix, hidden_dim=50, sigma_scope=25, sigma_prime=0.1, dropout=0.2):
        super().__init__()
        self.hidden_dim   = hidden_dim
        self.sigma_scope  = sigma_scope   # Propagation scope (paper: 15-35)
        self.sigma_prime  = sigma_prime   # Gaussian std dev (paper: 0.1)

        self.encoder      = SharedBLSTM(embed_matrix, hidden_dim, dropout)
        self.pos_attn     = PositionalAttention(2 * hidden_dim)

    def forward(self, q_ids, a_ids, q_len, a_len, q_pos):
        """
        Args:
            q_ids:  (B, L_q) question token ids
            a_ids:  (B, L_a) answer token ids
            q_len:  (B,) question lengths
            a_len:  (B,) answer lengths
            q_pos:  list of lists – positions in answer where question words appear

        Returns:
            sim: (B,) similarity scores in (0, 1]
        """
        # ── Encode ──
        _, q_pooled         = self.encoder(q_ids, q_len)       # (B, 2H)
        a_hidden, _         = self.encoder(a_ids, a_len)       # (B, L_a, 2H)

        # ── Position-aware influence (Gaussian kernel) ──
        d_hat = batch_position_influence(
            a_len, q_pos, a_ids.size(1),
            self.sigma_scope, self.sigma_prime
        ).to(a_hidden.device)  # (B, L_a)

        # ── Positional attention ──
        a_mask = (a_ids != 0).float()  # (B, L_a)
        a_attended = self.pos_attn(a_hidden, q_pooled, a_mask, d_hat)  # (B, 2H)

        # ── Manhattan distance similarity ──
        # sim(q, a) = exp(-||h_q - h_a||_1)
        sim = torch.exp(-torch.sum(torch.abs(q_pooled - a_attended), dim=1))  # (B,)
        return sim
