import os
import re
import urllib.request
import zipfile
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset
import nltk

try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception:
    pass

# ── Constants ──
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
GLOVE_URL = 'https://nlp.stanford.edu/data/glove.6B.zip'

def download_glove(glove_dir, glove_file, glove_zip):
    """Downloads and extracts GloVe embeddings if not present."""
    if not os.path.exists(glove_file):
        os.makedirs(glove_dir, exist_ok=True)
        print("Downloading GloVe... This may take a few minutes.")
        urllib.request.urlretrieve(GLOVE_URL, glove_zip)
        print("Extracting...")
        with zipfile.ZipFile(glove_zip, 'r') as zip_ref:
            zip_ref.extractall(glove_dir)
        print("Download complete.")

def load_glove(path, embed_dim=100):
    """Parses GloVe text file into a dictionary."""
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
    """Lowercases and tokenizes text."""
    text = text.lower()
    return [t for t in re.findall(r"\w+|[^\w\s]", text) if t.strip()]

def build_vocab(wiki_train, trec_train):
    """Builds a shared vocabulary dictionary from training sets."""
    counter = Counter()
    for dataset in [wiki_train, trec_train]:
        for sample in dataset:
            counter.update(tokenize(sample['question']))
            counter.update(tokenize(sample['answer']))

    vocab = [PAD_TOKEN, UNK_TOKEN] + list(counter.keys())
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word

def build_embedding_matrix(word2idx, glove_vectors, emb_dim=100):
    """Creates a PyTorch tensor matrix aligning vocabulary to GloVe vectors."""
    matrix = np.random.uniform(-0.25, 0.25, (len(word2idx), emb_dim)).astype(np.float32)
    matrix[0] = np.zeros(emb_dim) # PAD token is zero vector
    for word, idx in word2idx.items():
        if word in glove_vectors:
            matrix[idx] = glove_vectors[word]
    return torch.tensor(matrix, dtype=torch.float32)

def encode_and_pad(tokens, max_len, word2idx):
    """Converts tokens to indices, truncates, and pads to max_len."""
    encoded = [word2idx.get(t, word2idx[UNK_TOKEN]) for t in tokens]
    encoded = encoded[:max_len]
    length = len(encoded)
    encoded += [word2idx[PAD_TOKEN]] * (max_len - length)
    return encoded, length

def find_question_positions(q_tokens, a_tokens):
    """Finds positions in the answer where question words appear."""
    q_set = set(q_tokens)
    return [i for i, word in enumerate(a_tokens) if word in q_set]

class QADataset(Dataset):
    """PyTorch Dataset wrapper for Question Answering data."""
    def __init__(self, hf_split, word2idx, max_q_len=40, max_a_len=100, qid_field='question_id'):
        self.samples = []
        for ex in hf_split:
            q_tok = tokenize(ex['question'])
            a_tok = tokenize(ex['answer'])
            
            q_ids, q_len = encode_and_pad(q_tok, max_q_len, word2idx)
            a_ids, a_len = encode_and_pad(a_tok, max_a_len, word2idx)
            q_positions = find_question_positions(q_tok, a_tok[:max_a_len])
            
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
    """Custom collate function to batch dynamic lengths properly."""
    return {
        'q_ids':  torch.stack([b['q_ids'] for b in batch]),
        'a_ids':  torch.stack([b['a_ids'] for b in batch]),
        'q_len':  torch.tensor([b['q_len'] for b in batch]),
        'a_len':  torch.tensor([b['a_len'] for b in batch]),
        'q_pos':  [b['q_pos'] for b in batch],
        'label':  torch.stack([b['label'] for b in batch]),
        'qid':    [b['qid'] for b in batch],
    }