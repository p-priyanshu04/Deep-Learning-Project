import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from 3_model_implementation import SharedBLSTM 

# ── Baseline Models ──
class AvgPoolBLSTM(nn.Module):
    def __init__(self, embed_matrix, hidden_dim=50, dropout=0.2):
        super().__init__()
        self.encoder = SharedBLSTM(embed_matrix, hidden_dim, dropout)

    def forward(self, q_ids, a_ids, q_len, a_len, q_pos):
        _, q_pooled = self.encoder(q_ids, q_len)
        _, a_pooled = self.encoder(a_ids, a_len)
        sim = torch.exp(-torch.sum(torch.abs(q_pooled - a_pooled), dim=1))
        return sim


class AttentionBLSTM(nn.Module):
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

# ── Evaluation Metrics ──
def compute_metrics(qid_list, scores, labels):
    groups = defaultdict(list)
    for qid, score, label in zip(qid_list, scores, labels):
        groups[qid].append((score, label))

    avg_precisions, reciprocal_ranks = [], []

    for qid, pairs in groups.items():
        pairs.sort(key=lambda x: x[0], reverse=True)
        sorted_labels = [p[1] for p in pairs]

        if sum(sorted_labels) == 0:
            continue

        num_correct = 0
        precision_sum = 0.0
        for rank, lbl in enumerate(sorted_labels, 1):
            if lbl == 1:
                num_correct += 1
                precision_sum += num_correct / rank
        avg_precisions.append(precision_sum / num_correct)

        for rank, lbl in enumerate(sorted_labels, 1):
            if lbl == 1:
                reciprocal_ranks.append(1.0 / rank)
                break

    MAP = np.mean(avg_precisions) if avg_precisions else 0.0
    MRR = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    return MAP, MRR

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_scores, all_labels, all_qids = [], [], []
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        q_ids, a_ids = batch['q_ids'].to(device), batch['a_ids'].to(device)
        q_len, a_len = batch['q_len'].to(device), batch['a_len'].to(device)
        labels = batch['label'].to(device)

        sim = model(q_ids, a_ids, q_len, a_len, batch['q_pos'])
        sim = sim.clamp(1e-7, 1 - 1e-7)
        loss = F.binary_cross_entropy(sim, labels)
        
        total_loss += loss.item()
        n_batches += 1

        all_scores.extend(sim.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_qids.extend(batch['qid'])

    avg_loss = total_loss / max(n_batches, 1)
    MAP, MRR = compute_metrics(all_qids, np.array(all_scores), np.array(all_labels))
    return avg_loss, MAP, MRR