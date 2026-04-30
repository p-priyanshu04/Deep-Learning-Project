import torch
import torch.nn.functional as F
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Import classes and functions from the other scripts
from 2_baseline_metrics import AvgPoolBLSTM, AttentionBLSTM, evaluate
from 3_model_implementation import RNNPOA, compute_position_influence, batch_position_influence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HIDDEN_DIM  = 50
SIGMA_SCOPE = 25
SIGMA_PRIME = 0.1

# ── Extraction Functions for Heatmaps ──
def extract_attention_weights_avg(sample):
    actual_len = sample['a_len']
    alpha = np.ones(actual_len) / actual_len
    return alpha

@torch.no_grad()
def extract_attention_weights_att(model_att, sample):
    model_att.eval()
    q_ids = sample['q_ids'].unsqueeze(0).to(device)      
    a_ids = sample['a_ids'].unsqueeze(0).to(device)      
    q_len = torch.tensor([sample['q_len']]).to(device)    
    a_len = torch.tensor([sample['a_len']]).to(device)    

    _, q_pooled = model_att.encoder(q_ids, q_len)    
    a_hidden, _ = model_att.encoder(a_ids, a_len)    

    Wq = model_att.W(q_pooled)                       
    scores = torch.bmm(a_hidden, Wq.unsqueeze(2)).squeeze(2)  
    a_mask = (a_ids != 0).float()                    
    scores = scores.masked_fill(a_mask == 0, -1e9)
    alpha = F.softmax(scores, dim=1)                 

    return alpha[0, :sample['a_len']].cpu().numpy()

@torch.no_grad()
def extract_attention_weights_poa(model_poa, sample):
    model_poa.eval()
    q_ids = sample['q_ids'].unsqueeze(0).to(device)
    a_ids = sample['a_ids'].unsqueeze(0).to(device)
    q_len = torch.tensor([sample['q_len']]).to(device)
    a_len_t = torch.tensor([sample['a_len']]).to(device)
    q_pos = [sample['q_pos']]
    
    _, q_pooled = model_poa.encoder(q_ids, q_len)
    a_hidden, _ = model_poa.encoder(a_ids, a_len_t)
    
    d_hat = batch_position_influence(
        a_len_t, q_pos, a_ids.size(1), model_poa.sigma_scope, model_poa.sigma_prime
    ).to(device) 
    
    a_mask = (a_ids != 0).float()
    Wq = model_poa.pos_attn.W(q_pooled)
    scores = torch.bmm(a_hidden, Wq.unsqueeze(2)).squeeze(2)
    scores = scores.masked_fill(a_mask == 0, -1e9)
    exp_scores = torch.exp(scores - scores.max(dim=1, keepdim=True).values)
    exp_scores = exp_scores * a_mask
    modulated = exp_scores * (1.0 + d_hat)
    alpha = modulated / modulated.sum(dim=1, keepdim=True).clamp(min=1e-9)
    
    return (alpha[0, :sample['a_len']].cpu().numpy(), d_hat[0, :sample['a_len']].cpu().numpy())

# ── Heatmap Plotting Function ──
def plot_attention_comparison(sample, alpha_avg, alpha_att, alpha_poa, d_hat, idx2word, max_tokens=40, figsize=(22, 5)):
    actual_len = min(sample['a_len'], max_tokens)
    
    a_ids = sample['a_ids'].numpy()
    a_tokens = [idx2word.get(int(a_ids[i]), '<?>') for i in range(actual_len)]
    q_ids = sample['q_ids'].numpy()
    q_tokens = [idx2word.get(int(q_ids[i]), '<?>') for i in range(sample['q_len'])]
    q_text = ' '.join(q_tokens)
    
    q_pos_set = set(p for p in sample['q_pos'] if p < actual_len)
    
    alpha_avg_trim = alpha_avg[:actual_len]
    alpha_att_trim = alpha_att[:actual_len]
    alpha_poa_trim = alpha_poa[:actual_len]
    d_hat_trim = d_hat[:actual_len]
    
    a_labels = [f'★ {tok}' if i in q_pos_set else tok for i, tok in enumerate(a_tokens)]
    
    fig, axes = plt.subplots(1, 4, figsize=figsize, gridspec_kw={'width_ratios': [3, 3, 3, 1.5]})
    fig.suptitle(f'Q: {q_text}', fontsize=12, fontweight='bold', wrap=True, y=1.05)
    
    ax = axes[0]
    im0 = ax.imshow(alpha_avg_trim.reshape(1, -1), cmap='YlOrRd', aspect='auto', vmin=0)
    ax.set_yticks([]); ax.set_xticks(range(actual_len)); ax.set_xticklabels(a_labels, rotation=90, fontsize=9)
    ax.set_title('RNN-AVG (Uniform Pooling)', fontsize=10, fontweight='bold')
    plt.colorbar(im0, ax=ax, shrink=0.3, label='Weight')

    ax = axes[1]
    im1 = ax.imshow(alpha_att_trim.reshape(1, -1), cmap='YlOrRd', aspect='auto', vmin=0)
    ax.set_yticks([]); ax.set_xticks(range(actual_len)); ax.set_xticklabels(a_labels, rotation=90, fontsize=9)
    ax.set_title('RNN-ATT (Classical Attention)', fontsize=10, fontweight='bold')
    plt.colorbar(im1, ax=ax, shrink=0.3, label='α')
    
    ax = axes[2]
    im2 = ax.imshow(alpha_poa_trim.reshape(1, -1), cmap='YlOrRd', aspect='auto', vmin=0)
    ax.set_yticks([]); ax.set_xticks(range(actual_len)); ax.set_xticklabels(a_labels, rotation=90, fontsize=9)
    ax.set_title('RNN-POA (Positional Attention)', fontsize=10, fontweight='bold')
    plt.colorbar(im2, ax=ax, shrink=0.3, label='α̃')
    
    ax = axes[3]
    colors = ['#e74c3c' if i in q_pos_set else '#3498db' for i in range(actual_len)]
    ax.barh(range(actual_len), d_hat_trim, color=colors, height=0.7)
    ax.set_yticks(range(actual_len)); ax.set_yticklabels(a_labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('d̂ (influence)', fontsize=10)
    ax.set_title('Position Influence', fontsize=10, fontweight='bold')
    ax.axvline(x=0, color='gray', linewidth=0.5)
    
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor='#e74c3c', label='Q-word match'), Patch(facecolor='#3498db', label='No match')], 
              fontsize=8, loc='lower right')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("Loading models and data for evaluation...")
    
    embedding_matrix = torch.load('embedding_matrix.pt', map_location=device)
    dataloaders = torch.load('dataloaders.pt')
    with open('vocab.pkl', 'rb') as f:
        idx2word = pickle.load(f)['idx2word']
    with open('training_histories.pkl', 'rb') as f:
        histories = pickle.load(f)

    # 1. Load Trained Models
    wiki_model_poa = RNNPOA(embedding_matrix, hidden_dim=HIDDEN_DIM, sigma_scope=SIGMA_SCOPE, sigma_prime=SIGMA_PRIME).to(device)
    wiki_model_poa.load_state_dict(torch.load('best_wiki_poa.pt', map_location=device, weights_only=True))

    wiki_model_base = AttentionBLSTM(embedding_matrix, hidden_dim=HIDDEN_DIM).to(device)
    wiki_model_base.load_state_dict(torch.load('best_wiki_base.pt', map_location=device, weights_only=True))

    wiki_model_avg = AvgPoolBLSTM(embedding_matrix, hidden_dim=HIDDEN_DIM).to(device)
    wiki_model_avg.load_state_dict(torch.load('best_wiki_avg.pt', map_location=device, weights_only=True))

    trec_model_poa = RNNPOA(embedding_matrix, hidden_dim=HIDDEN_DIM, sigma_scope=SIGMA_SCOPE, sigma_prime=SIGMA_PRIME).to(device)
    trec_model_poa.load_state_dict(torch.load('best_trec_poa.pt', map_location=device, weights_only=True))

    trec_model_base = AttentionBLSTM(embedding_matrix, hidden_dim=HIDDEN_DIM).to(device)
    trec_model_base.load_state_dict(torch.load('best_trec_base.pt', map_location=device, weights_only=True))

    trec_model_avg = AvgPoolBLSTM(embedding_matrix, hidden_dim=HIDDEN_DIM).to(device)
    trec_model_avg.load_state_dict(torch.load('best_trec_avg.pt', map_location=device, weights_only=True))

    # 2. Evaluate on Test Sets
    _, wiki_test_map_poa, wiki_test_mrr_poa = evaluate(wiki_model_poa, dataloaders['wiki_test'], device)
    _, wiki_test_map_base, wiki_test_mrr_base = evaluate(wiki_model_base, dataloaders['wiki_test'], device)
    _, wiki_test_map_avg, wiki_test_mrr_avg = evaluate(wiki_model_avg, dataloaders['wiki_test'], device)

    _, trec_test_map_poa, trec_test_mrr_poa = evaluate(trec_model_poa, dataloaders['trec_test'], device)
    _, trec_test_map_base, trec_test_mrr_base = evaluate(trec_model_base, dataloaders['trec_test'], device)
    _, trec_test_map_avg, trec_test_mrr_avg = evaluate(trec_model_avg, dataloaders['trec_test'], device)

    print('\n' + '='*70)
    print('  COMPREHENSIVE TEST SET RESULTS')
    print('='*70)
    print(f'  {"Model":<30} {"WikiQA MAP":>10} {"TREC-QA MAP":>10}')
    print('  ' + '-'*55)
    print(f'  {"RNN-AVG (avg pool)":<30} {wiki_test_map_avg:>10.4f} {trec_test_map_avg:>10.4f}')
    print(f'  {"RNN-ATT (attention)":<30} {wiki_test_map_base:>10.4f} {trec_test_map_base:>10.4f}')
    print(f'  {"RNN-POA (positional attn)":<30} {wiki_test_map_poa:>10.4f} {trec_test_map_poa:>10.4f}')
    print('='*70 + '\n')

    # 3. Plot Training Curves
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Training Curves: WikiQA vs TREC-QA', fontsize=16, fontweight='bold')

    datasets_info = [
        ('WikiQA', histories['wiki_poa'], histories['wiki_base'], histories['wiki_avg'], 0),
        ('TREC-QA', histories['trec_poa'], histories['trec_base'], histories['trec_avg'], 1),
    ]

    for ds_name, hist_poa, hist_base, hist_avg, row in datasets_info:
        epochs_poa  = range(1, len(hist_poa['train_loss']) + 1)
        epochs_base = range(1, len(hist_base['train_loss']) + 1)
        epochs_avg  = range(1, len(hist_avg['train_loss']) + 1)

        ax = axes[row][0]
        ax.plot(epochs_poa,  hist_poa['dev_loss'],    'b-', label='RNN-POA')
        ax.plot(epochs_base, hist_base['dev_loss'],   'r-', label='RNN-ATT')
        ax.plot(epochs_avg,  hist_avg['dev_loss'],    'g-', label='RNN-AVG')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Dev Loss')
        ax.set_title(f'{ds_name} — Validation Loss')
        ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[row][1]
        ax.plot(epochs_poa,  hist_poa['dev_map'],  'b-o', markersize=4, label='RNN-POA')
        ax.plot(epochs_base, hist_base['dev_map'], 'r-s', markersize=4, label='RNN-ATT')
        ax.plot(epochs_avg,  hist_avg['dev_map'],  'g-^', markersize=4, label='RNN-AVG')
        ax.set_xlabel('Epoch'); ax.set_ylabel('MAP')
        ax.set_title(f'{ds_name} — Dev MAP')
        ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[row][2]
        ax.plot(epochs_poa,  hist_poa['dev_mrr'],  'b-o', markersize=4, label='RNN-POA')
        ax.plot(epochs_base, hist_base['dev_mrr'], 'r-s', markersize=4, label='RNN-ATT')
        ax.plot(epochs_avg,  hist_avg['dev_mrr'],  'g-^', markersize=4, label='RNN-AVG')
        ax.set_xlabel('Epoch'); ax.set_ylabel('MRR')
        ax.set_title(f'{ds_name} — Dev MRR')
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('training_curves.png')
    print("Saved training_curves.png")

    # 4. Generate Heatmaps for Examples
    print("Generating heatmaps...")
    wiki_dev_ds = dataloaders['wiki_dev'].dataset
    viz_candidates = []
    for i, sample in enumerate(wiki_dev_ds.samples):
        if sample['label'].item() == 1.0 and len(sample['q_pos']) >= 2 and 8 <= sample['a_len'] <= 35:
            viz_candidates.append(i)

    selected_indices = viz_candidates[:4]
    
    for idx_num, sample_idx in enumerate(selected_indices):
        sample = wiki_dev_ds.samples[sample_idx]
        alpha_avg = extract_attention_weights_avg(sample)
        alpha_att = extract_attention_weights_att(wiki_model_base, sample)
        alpha_poa, d_hat = extract_attention_weights_poa(wiki_model_poa, sample)
        
        fig = plot_attention_comparison(sample, alpha_avg, alpha_att, alpha_poa, d_hat, idx2word)
        fname = f'attention_heatmap_{idx_num+1}.png'
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"Saved {fname}")
        
    print("Visualization complete.")