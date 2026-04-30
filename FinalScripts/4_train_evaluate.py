import torch
import torch.nn.functional as F
import pickle
from tqdm.auto import tqdm
import random
import numpy as np

# Import classes and functions from the other scripts
from 2_baseline_metrics import AvgPoolBLSTM, AttentionBLSTM, evaluate
from 3_model_implementation import RNNPOA

# ── Constants ──
HIDDEN_DIM  = 50
SIGMA_SCOPE = 25
SIGMA_PRIME = 0.1
N_EPOCHS    = 30
PATIENCE    = 5
LR          = 1.0
SEED        = 42

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, train_loader, dev_loader, n_epochs, patience, lr, save_name):
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
            q_ids, a_ids = batch['q_ids'].to(device), batch['a_ids'].to(device)
            q_len, a_len = batch['q_len'].to(device), batch['a_len'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            sim = model(q_ids, a_ids, q_len, a_len, batch['q_pos'])
            sim = sim.clamp(1e-7, 1 - 1e-7)

            loss = F.binary_cross_entropy(sim, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        dev_loss, dev_map, dev_mrr = evaluate(model, dev_loader, device)

        history['train_loss'].append(avg_train_loss)
        history['dev_loss'].append(dev_loss)
        history['dev_map'].append(dev_map)
        history['dev_mrr'].append(dev_mrr)

        print(f'Epoch {epoch:2d} | Train Loss: {avg_train_loss:.4f} | Dev Loss: {dev_loss:.4f} | Dev MAP: {dev_map:.4f} | Dev MRR: {dev_mrr:.4f}')

        if dev_map > best_dev_map:
            best_dev_map = dev_map
            patience_counter = 0
            torch.save(model.state_dict(), save_name)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

    model.load_state_dict(torch.load(save_name, weights_only=True))
    return history

if __name__ == "__main__":
    # 1. Load Preprocessed Data
    print("Loading preprocessed data...")
    embedding_matrix = torch.load('embedding_matrix.pt')
    dataloaders = torch.load('dataloaders.pt')

    # 2. Train WikiQA Models
    print("\n--- Training WikiQA Models ---")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    
    wiki_model_poa = RNNPOA(embedding_matrix, hidden_dim=HIDDEN_DIM, sigma_scope=SIGMA_SCOPE, sigma_prime=SIGMA_PRIME).to(device)
    wiki_history_poa = train_model(wiki_model_poa, dataloaders['wiki_train'], dataloaders['wiki_dev'], N_EPOCHS, PATIENCE, LR, 'best_wiki_poa.pt')

    wiki_model_base = AttentionBLSTM(embedding_matrix, hidden_dim=HIDDEN_DIM).to(device)
    wiki_history_base = train_model(wiki_model_base, dataloaders['wiki_train'], dataloaders['wiki_dev'], N_EPOCHS, PATIENCE, LR, 'best_wiki_base.pt')

    wiki_model_avg = AvgPoolBLSTM(embedding_matrix, hidden_dim=HIDDEN_DIM).to(device)
    wiki_history_avg = train_model(wiki_model_avg, dataloaders['wiki_train'], dataloaders['wiki_dev'], N_EPOCHS, PATIENCE, LR, 'best_wiki_avg.pt')

    # 3. Train TREC-QA Models
    print("\n--- Training TREC-QA Models ---")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    trec_model_poa = RNNPOA(embedding_matrix, hidden_dim=HIDDEN_DIM, sigma_scope=SIGMA_SCOPE, sigma_prime=SIGMA_PRIME).to(device)
    trec_history_poa = train_model(trec_model_poa, dataloaders['trec_train'], dataloaders['trec_dev'], N_EPOCHS, PATIENCE, LR, 'best_trec_poa.pt')

    trec_model_base = AttentionBLSTM(embedding_matrix, hidden_dim=HIDDEN_DIM).to(device)
    trec_history_base = train_model(trec_model_base, dataloaders['trec_train'], dataloaders['trec_dev'], N_EPOCHS, PATIENCE, LR, 'best_trec_base.pt')

    trec_model_avg = AvgPoolBLSTM(embedding_matrix, hidden_dim=HIDDEN_DIM).to(device)
    trec_history_avg = train_model(trec_model_avg, dataloaders['trec_train'], dataloaders['trec_dev'], N_EPOCHS, PATIENCE, LR, 'best_trec_avg.pt')

    # 4. Save Histories
    histories = {
        'wiki_poa': wiki_history_poa, 'wiki_base': wiki_history_base, 'wiki_avg': wiki_history_avg,
        'trec_poa': trec_history_poa, 'trec_base': trec_history_base, 'trec_avg': trec_history_avg
    }
    with open('training_histories.pkl', 'wb') as f:
        pickle.dump(histories, f)

    print("Training complete. Weights and histories saved.")