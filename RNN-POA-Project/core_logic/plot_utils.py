import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from core_logic.model_architectures import batch_position_influence

def extract_attention_weights_avg(sample):
    """Extracts 'pseudo' attention weights for RNN-AVG (Uniform distribution)."""
    actual_len = sample['a_len']
    return np.ones(actual_len) / actual_len

@torch.no_grad()
def extract_attention_weights_att(model_att, sample, device):
    """Extracts classical attention weights from the AttentionBLSTM model."""
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
def extract_attention_weights_poa(model_poa, sample, device):
    """Extracts positional attention weights and influence vector from the RNNPOA model."""
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

def plot_attention_comparison(sample, alpha_avg, alpha_att, alpha_poa, d_hat, idx2word, max_tokens=40, figsize=(22, 5)):
    """Plots side-by-side attention heatmaps for all 3 models."""
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