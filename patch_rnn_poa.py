import json
import os

NOTEBOOK_PATH = 'rnn-poa.ipynb'
BACKUP_PATH = 'rnn-poa.ipynb.backup'

def patch_notebook():
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    # Backup
    with open(BACKUP_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

    cells = nb['cells']
    
    # 1. Insert AttentionBLSTM class before RNNPOA
    rnn_poa_idx = -1
    for i, c in enumerate(cells):
        if c['cell_type'] == 'code':
            src = c['source']
            if isinstance(src, list):
                src = ''.join(src)
            if 'class RNNPOA' in src:
                rnn_poa_idx = i
                break
            
    if rnn_poa_idx != -1:
        attn_blstm_code = [
            "class AttentionBLSTM(nn.Module):\n",
            "    \"\"\"\n",
            "    Baseline: classical attention-based BLSTM (no positional influence).\n",
            "    Used for comparison to show the benefit of positional attention.\n",
            "    \"\"\"\n",
            "    def __init__(self, embed_matrix, hidden_dim=50, dropout=0.2):\n",
            "        super().__init__()\n",
            "        self.encoder = SharedBLSTM(embed_matrix, hidden_dim, dropout)\n",
            "        self.W = nn.Linear(2*hidden_dim, 2*hidden_dim, bias=False)\n",
            "\n",
            "    def forward(self, q_ids, a_ids, q_len, a_len, q_pos):\n",
            "        _, q_pooled   = self.encoder(q_ids, q_len)\n",
            "        a_hidden, _   = self.encoder(a_ids, a_len)\n",
            "\n",
            "        Wq = self.W(q_pooled)\n",
            "        scores = torch.bmm(a_hidden, Wq.unsqueeze(2)).squeeze(2)\n",
            "        a_mask = (a_ids != 0).float()\n",
            "        scores = scores.masked_fill(a_mask == 0, -1e9)\n",
            "        alpha  = F.softmax(scores, dim=1)\n",
            "        a_attended = torch.bmm(alpha.unsqueeze(1), a_hidden).squeeze(1)\n",
            "\n",
            "        sim = torch.exp(-torch.sum(torch.abs(q_pooled - a_attended), dim=1))\n",
            "        return sim\n"
        ]
        
        new_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": attn_blstm_code
        }
        cells.insert(rnn_poa_idx, new_cell)
        print(f"Inserted AttentionBLSTM at index {rnn_poa_idx}")

    # 2. Add wiki_model_base training
    training_idx = -1
    for i, c in enumerate(cells):
        if c['cell_type'] == 'code':
            src = c['source']
            if isinstance(src, list):
                src = ''.join(src)
            if 'wiki_model_poa = RNNPOA' in src:
                training_idx = i
                break
            
    if training_idx != -1:
        src = cells[training_idx]['source']
        if isinstance(src, str):
            src_lines = [line + '\n' for line in src.split('\n')]
            # correct the last element if it shouldn't have \n
            if src_lines and not src.endswith('\n'):
                src_lines[-1] = src_lines[-1][:-1]
        else:
            src_lines = src
            
        insert_line_idx = -1
        for i, line in enumerate(src_lines):
            if 'save_name=\'best_wiki_poa.pt\'' in line:
                insert_line_idx = i + 1
                break
                
        if insert_line_idx != -1:
            wiki_base_code = [
                "\n",
                "wiki_model_base = AttentionBLSTM(embedding_matrix, hidden_dim=HIDDEN_DIM).to(device)\n",
                "wiki_history_base = train_model(wiki_model_base, wiki_train_loader, wiki_dev_loader,\n",
                "    N_EPOCHS, PATIENCE, LR, save_name='best_wiki_base.pt')\n"
            ]
            cells[training_idx]['source'] = ''.join(src_lines[:insert_line_idx] + wiki_base_code + src_lines[insert_line_idx:])
            print(f"Inserted wiki_model_base training in cell {training_idx}")
            
    # Save back
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook successfully patched.")

if __name__ == '__main__':
    patch_notebook()
