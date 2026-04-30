import json
import matplotlib.pyplot as plt
import numpy as np

with open('../results/experiment_results.json', 'r') as f:
    results = json.load(f)

# ---

models = ['RNN-AVG', 'RNN-ATT', 'RNN-POA (Best)']
maps = [results['AVG']['MAP'], results['ATT']['MAP'], results['POA_Best']['MAP']]
mrrs = [results['AVG']['MRR'], results['ATT']['MRR'], results['POA_Best']['MRR']]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width/2, maps, width, label='MAP', color='skyblue')
bars2 = ax.bar(x + width/2, mrrs, width, label='MRR', color='salmon')

ax.set_ylabel('Scores')
ax.set_title('Test Set Performance: AVG vs ATT vs POA')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# ---

paper_map, paper_mrr = 0.7212, 0.7312
our_map, our_mrr = results['POA_Best']['MAP'], results['POA_Best']['MRR']

x = np.arange(2)
width = 0.35

fig, ax = plt.subplots(figsize=(6, 5))
bars1 = ax.bar(x - width/2, [paper_map, our_map], width, label='MAP', color='mediumpurple')
bars2 = ax.bar(x + width/2, [paper_mrr, our_mrr], width, label='MRR', color='lightgreen')

ax.set_ylabel('Scores')
ax.set_title('RNN-POA: Paper vs Ours (WikiQA)')
ax.set_xticks(x)
ax.set_xticklabels(['Paper', 'Ours'])
ax.legend()

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# ---

sigmas = sorted([int(k) for k in results['POA_Tuning'].keys()])
sig_maps = [results['POA_Tuning'][str(s)]['MAP'] for s in sigmas]
sig_mrrs = [results['POA_Tuning'][str(s)]['MRR'] for s in sigmas]

plt.figure(figsize=(8, 5))
plt.plot(sigmas, sig_maps, marker='o', label='MAP', color='blue')
plt.plot(sigmas, sig_mrrs, marker='s', label='MRR', color='red')
plt.xlabel('Sigma (Propagation Scope)')
plt.ylabel('Score')
plt.title('Impact of Sigma on RNN-POA Performance')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ---

