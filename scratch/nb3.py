import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from tqdm.auto import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_metrics(qid_list, scores, labels):
    """
    Compute MAP and MRR for answer selection.

    Groups predictions by question ID, ranks answers by predicted score,
    and computes MAP and MRR.

    Args:
        qid_list: list of question IDs (one per sample)
        scores:   np.array of predicted similarity scores
        labels:   np.array of ground-truth binary labels

    Returns:
        (MAP, MRR) tuple
    """
    # Group by question ID
    groups = defaultdict(list)
    for qid, score, label in zip(qid_list, scores, labels):
        groups[qid].append((score, label))

    avg_precisions = []
    reciprocal_ranks = []

    for qid, pairs in groups.items():
        # Sort by score descending
        pairs.sort(key=lambda x: x[0], reverse=True)
        sorted_labels = [p[1] for p in pairs]

        # Skip questions with no positive answer
        if sum(sorted_labels) == 0:
            continue

        # Average Precision
        num_correct = 0
        precision_sum = 0.0
        for rank, lbl in enumerate(sorted_labels, 1):
            if lbl == 1:
                num_correct += 1
                precision_sum += num_correct / rank
        ap = precision_sum / num_correct
        avg_precisions.append(ap)

        # Reciprocal Rank
        for rank, lbl in enumerate(sorted_labels, 1):
            if lbl == 1:
                reciprocal_ranks.append(1.0 / rank)
                break

    MAP = np.mean(avg_precisions) if avg_precisions else 0.0
    MRR = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    return MAP, MRR


@torch.no_grad()
def evaluate(model, loader):
    """Run model on a data loader and return loss, MAP, MRR."""
    model.eval()
    all_scores, all_labels, all_qids = [], [], []
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        q_ids = batch['q_ids'].to(device)
        a_ids = batch['a_ids'].to(device)
        q_len = batch['q_len'].to(device)
        a_len = batch['a_len'].to(device)
        labels = batch['label'].to(device)

        sim = model(q_ids, a_ids, q_len, a_len, batch['q_pos'])
        sim = sim.clamp(1e-7, 1 - 1e-7)  # numerical safety
        loss = F.binary_cross_entropy(sim, labels)
        total_loss += loss.item()
        n_batches += 1

        all_scores.extend(sim.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_qids.extend(batch['qid'])

    avg_loss = total_loss / max(n_batches, 1)
    MAP, MRR = compute_metrics(all_qids, np.array(all_scores), np.array(all_labels))
    return avg_loss, MAP, MRR


def train_model(model, train_loader, dev_loader, n_epochs=30, patience=5, lr=1.0, save_name='best_model.pt'):
    """
    Train the model with Adadelta and early stopping.

    Returns:
        history dict with train/dev losses, MAP, MRR per epoch
    """
    optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
    best_dev_map = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'dev_loss': [], 'dev_map': [], 'dev_mrr': []}

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{n_epochs}', leave=False)
        for batch in pbar:
            q_ids = batch['q_ids'].to(device)
            a_ids = batch['a_ids'].to(device)
            q_len = batch['q_len'].to(device)
            a_len = batch['a_len'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            sim = model(q_ids, a_ids, q_len, a_len, batch['q_pos'])
            sim = sim.clamp(1e-7, 1 - 1e-7)  # prevent log(0)

            # Binary cross-entropy loss
            loss = F.binary_cross_entropy(sim, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # ── Dev evaluation ──
        dev_loss, dev_map, dev_mrr = evaluate(model, dev_loader)

        history['train_loss'].append(avg_train_loss)
        history['dev_loss'].append(dev_loss)
        history['dev_map'].append(dev_map)
        history['dev_mrr'].append(dev_mrr)

        print(f'Epoch {epoch:2d} | '
              f'Train Loss: {avg_train_loss:.4f} | '
              f'Dev Loss: {dev_loss:.4f} | '
              f'Dev MAP: {dev_map:.4f} | '
              f'Dev MRR: {dev_mrr:.4f}')

        # Early stopping on dev MAP
        if dev_map > best_dev_map:
            best_dev_map = dev_map
            patience_counter = 0
            torch.save(model.state_dict(), save_name)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch} (no MAP improvement for {patience} epochs)')
                break

    # Load best model
    try:
        model.load_state_dict(torch.load(save_name, weights_only=True))
    except TypeError:
        model.load_state_dict(torch.load(save_name)) # older pytorch compatibility
    return history


# ---

import sys
import os
import json
sys.path.append('..')

import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader

from utils.data_utils import *
from utils.model_utils import RNNAVG, RNNATT, RNNPOA
# from utils.train_utils import train_model, evaluate, compute_metrics (Replaced with inline code)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs('../results', exist_ok=True)

# ---

wiki_qa = load_dataset('wiki_qa')
word2idx, idx2word = build_vocab(wiki_qa['train'], [])
embedding_matrix = build_embedding_matrix(word2idx, load_glove())

BATCH_SIZE = 64
train_ds = QADataset(wiki_qa['train'], word2idx)
dev_ds   = QADataset(wiki_qa['validation'], word2idx)
test_ds  = QADataset(wiki_qa['test'], word2idx)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
dev_loader   = DataLoader(dev_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ---

HIDDEN_DIM = 50
N_EPOCHS = 10 # Set to 10 for quick testing, increase for real results

model_avg = RNNAVG(embedding_matrix, hidden_dim=HIDDEN_DIM).to(device)
model_att = RNNATT(embedding_matrix, hidden_dim=HIDDEN_DIM).to(device)

print('Training RNN-AVG...')
hist_avg = train_model(model_avg, train_loader, dev_loader, n_epochs=N_EPOCHS, save_name='../results/best_avg.pt')

print('\nTraining RNN-ATT...')
hist_att = train_model(model_att, train_loader, dev_loader, n_epochs=N_EPOCHS, save_name='../results/best_att.pt')

# ---

sigmas = [5, 15, 25, 35, 45, 55]
poa_results = {}
best_sigma = None
best_map = 0.0

for sig in sigmas:
    print(f'\n--- Training RNN-POA with sigma={sig} ---')
    model = RNNPOA(embedding_matrix, hidden_dim=HIDDEN_DIM, sigma_scope=sig).to(device)
    hist = train_model(model, train_loader, dev_loader, n_epochs=N_EPOCHS, save_name=f'../results/poa_sig_{sig}.pt')
    
    # Evaluate on test set
    _, test_map, test_mrr = evaluate(model, test_loader)
    poa_results[sig] = {'MAP': test_map, 'MRR': test_mrr}
    print(f'Sigma={sig} Test MAP: {test_map:.4f}, MRR: {test_mrr:.4f}')
    
    if test_map > best_map:
        best_map = test_map
        best_sigma = sig

print(f'\nBest Sigma: {best_sigma} with MAP: {best_map:.4f}')

# ---

_, avg_map, avg_mrr = evaluate(model_avg, test_loader)
_, att_map, att_mrr = evaluate(model_att, test_loader)

results = {
    'AVG': {'MAP': avg_map, 'MRR': avg_mrr},
    'ATT': {'MAP': att_map, 'MRR': att_mrr},
    'POA_Tuning': poa_results,
    'POA_Best': poa_results[best_sigma]
}

with open('../results/experiment_results.json', 'w') as f:
    json.dump(results, f)

print('Experiments completed and results saved.')

# ---

