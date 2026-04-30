import os
import re
import zipfile
import urllib.request
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import nltk

try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception:
    pass

from datasets import load_dataset

# ── Constants and Setup ──
GLOVE_DIR = 'glove_data'
GLOVE_FILE = os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')
GLOVE_ZIP  = os.path.join(GLOVE_DIR, 'glove.6B.zip')
GLOVE_URL  = 'https://nlp.stanford.edu/data/glove.6B.zip'
EMBED_DIM  = 100

PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'

MAX_Q_LEN = 40
MAX_A_LEN = 100
BATCH_SIZE = 64

# ── Functions ──
def download_glove():
    if not os.path.exists(GLOVE_FILE):
        os.makedirs(GLOVE_DIR, exist_ok=True)
        print("Downloading GloVe... This may take a few minutes.")
        urllib.request.urlretrieve(GLOVE_URL, GLOVE_ZIP)
        print("Extracting...")
        with zipfile.ZipFile(GLOVE_ZIP, 'r') as zip_ref:
            zip_ref.extractall(GLOVE_DIR)
        print("Download complete.")

def load_glove(path=GLOVE_FILE, embed_dim=EMBED_DIM):
    glove = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            if len(vector) == embed_dim:
                glove[word] = vector
    print(f"Loaded {len(glove)} GloVe vectors.")
    return glove

def tokenize(text):
    text = text.lower()
    return [t for t in re.findall(r"\w+|[^\w\s]", text) if t.strip()]

def build_vocab(wiki_train, trec_train):
    counter = Counter()
    for dataset in [wiki_train, trec_train]:
        for sample in dataset:
            counter.update(tokenize(sample['question']))
            counter.update(tokenize(sample['answer']))

    vocab = [PAD_TOKEN, UNK_TOKEN] + list(counter.keys())
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word

def build_embedding_matrix(word2idx, glove_vectors, emb_dim=EMBED_DIM):
    matrix = np.random.uniform(-0.25, 0.25, (len(word2idx), emb_dim)).astype(np.float32)
    matrix[0] = np.zeros(emb_dim) # PAD
    for word, idx in word2idx.items():
        if word in glove_vectors:
            matrix[idx] = glove_vectors[word]
    return torch.tensor(matrix, dtype=torch.float32)

def encode_and_pad(tokens, max_len, word2idx):
    encoded = [word2idx.get(t, word2idx[UNK_TOKEN]) for t in tokens]
    encoded = encoded[:max_len]
    length = len(encoded)
    encoded += [word2idx[PAD_TOKEN]] * (max_len - length)
    return encoded, length

def find_question_positions(q_tokens, a_tokens):
    q_set = set(q_tokens)
    return [i for i, word in enumerate(a_tokens) if word in q_set]

# ── Dataset Class ──
class QADataset(Dataset):
    def __init__(self, hf_split, word2idx, qid_field='question_id'):
        self.samples = []
        for ex in hf_split:
            q_tok = tokenize(ex['question'])
            a_tok = tokenize(ex['answer'])
            
            q_ids, q_len = encode_and_pad(q_tok, MAX_Q_LEN, word2idx)
            a_ids, a_len = encode_and_pad(a_tok, MAX_A_LEN, word2idx)
            q_positions = find_question_positions(q_tok, a_tok[:MAX_A_LEN])
            
            qid = ex[qid_field] if qid_field in ex else hash(ex['question']) % (10**8)

            self.samples.append({
                'q_ids':  torch.tensor(q_ids,  dtype=torch.long),
                'a_ids':  torch.tensor(a_ids,  dtype=torch.long),
                'q_len':  q_len,
                'a_len':  a_len,
                'q_pos':  q_positions,
                'label':  torch.tensor(ex['label'], dtype=torch.float32),
                'qid':    qid,
            })

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def collate_fn(batch):
    return {
        'q_ids':  torch.stack([b['q_ids'] for b in batch]),
        'a_ids':  torch.stack([b['a_ids'] for b in batch]),
        'q_len':  torch.tensor([b['q_len'] for b in batch]),
        'a_len':  torch.tensor([b['a_len'] for b in batch]),
        'q_pos':  [b['q_pos'] for b in batch],
        'label':  torch.stack([b['label'] for b in batch]),
        'qid':    [b['qid'] for b in batch],
    }

# ── Execution ──
if __name__ == "__main__":
    print("Loading datasets from HuggingFace...")
    wiki_qa = load_dataset('wiki_qa')
    trec_qa_raw = load_dataset('lucadiliello/trecqa')

    trec_dev_split = 'dev_clean' if 'dev_clean' in trec_qa_raw else 'validation'
    trec_test_split = 'test_clean' if 'test_clean' in trec_qa_raw else 'test'

    print("Building vocabulary...")
    word2idx, idx2word = build_vocab(wiki_qa['train'], trec_qa_raw['train'])
    print(f'Vocabulary size: {len(word2idx)}')

    download_glove()
    glove_vectors = load_glove()
    embedding_matrix = build_embedding_matrix(word2idx, glove_vectors)

    print("Creating datasets and dataloaders...")
    dataloaders = {
        'wiki_train': DataLoader(QADataset(wiki_qa['train'], word2idx), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn),
        'wiki_dev':   DataLoader(QADataset(wiki_qa['validation'], word2idx), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn),
        'wiki_test':  DataLoader(QADataset(wiki_qa['test'], word2idx), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn),
        'trec_train': DataLoader(QADataset(trec_qa_raw['train'], word2idx), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn),
        'trec_dev':   DataLoader(QADataset(trec_qa_raw[trec_dev_split], word2idx), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn),
        'trec_test':  DataLoader(QADataset(trec_qa_raw[trec_test_split], word2idx), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    }

    print("Saving preprocessed data to disk...")
    torch.save(embedding_matrix, 'embedding_matrix.pt')
    with open('vocab.pkl', 'wb') as f:
        pickle.dump({'word2idx': word2idx, 'idx2word': idx2word}, f)
    torch.save(dataloaders, 'dataloaders.pt')

    print("Preprocessing complete. Data saved.")