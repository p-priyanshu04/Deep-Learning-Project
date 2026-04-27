import os
import re
import zipfile
import urllib.request
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nltk

# Try to download NLTK tokenizer data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception:
    pass

GLOVE_DIR = 'glove_data'
GLOVE_FILE = os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')
GLOVE_ZIP  = os.path.join(GLOVE_DIR, 'glove.6B.zip')
GLOVE_URL  = 'https://nlp.stanford.edu/data/glove.6B.zip'
EMBED_DIM  = 100

PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'

MAX_Q_LEN = 40   # max question length (tokens)
MAX_A_LEN = 100  # max answer length (tokens)


def download_glove():
    """Download and extract GloVe embeddings if not present."""
    os.makedirs(GLOVE_DIR, exist_ok=True)
    if not os.path.exists(GLOVE_FILE):
        if not os.path.exists(GLOVE_ZIP):
            print('Downloading GloVe (this may take a few minutes)...')
            urllib.request.urlretrieve(GLOVE_URL, GLOVE_ZIP)
        print('Extracting...')
        with zipfile.ZipFile(GLOVE_ZIP, 'r') as z:
            z.extract('glove.6B.100d.txt', GLOVE_DIR)
        print('Done.')
    else:
        print('GloVe file already exists.')


def load_glove(path=GLOVE_FILE, embed_dim=EMBED_DIM):
    """Parse GloVe text file into {word: vector} dict."""
    embeddings = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip().split(' ')
            word = parts[0]
            vec  = np.array(parts[1:], dtype=np.float32)
            if len(vec) == embed_dim:
                embeddings[word] = vec
    print(f'Loaded {len(embeddings):,} GloVe vectors.')
    return embeddings


def tokenize(text):
    """
    Lowercase and tokenize text.
    Tries NLTK word_tokenize first; falls back to regex-based
    splitting if NLTK punkt data is unavailable.
    """
    text = text.lower()
    try:
        from nltk.tokenize import word_tokenize
        return word_tokenize(text)
    except LookupError:
        # Fallback: split on non-alphanumeric, keeping tokens
        return [t for t in re.findall(r"\w+|[^\w\s]", text) if t.strip()]


def build_vocab(wiki_train, trec_train):
    """Build vocabulary from BOTH training sets."""
    word_counts = Counter()

    for ex in wiki_train:
        word_counts.update(tokenize(ex['question']))
        word_counts.update(tokenize(ex['answer']))

    for ex in trec_train:
        word_counts.update(tokenize(ex['question']))
        word_counts.update(tokenize(ex['answer']))

    word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for w, _ in word_counts.most_common():
        word2idx[w] = len(word2idx)
    
    idx2word = {v: k for k, v in word2idx.items()}
    return word2idx, idx2word


def build_embedding_matrix(word2idx, glove_vectors, embed_dim=EMBED_DIM):
    """Build embedding matrix aligned to our vocabulary."""
    vocab_size = len(word2idx)
    embedding_matrix = np.random.uniform(-0.25, 0.25, (vocab_size, embed_dim)).astype(np.float32)
    embedding_matrix[0] = np.zeros(embed_dim)  # PAD = zero vector

    found = 0
    for word, idx in word2idx.items():
        if word in glove_vectors:
            embedding_matrix[idx] = glove_vectors[word]
            found += 1
    print(f'GloVe coverage: {found}/{vocab_size} ({100*found/vocab_size:.1f}%)')
    return embedding_matrix


def encode_and_pad(tokens, max_len, word2idx):
    """Convert tokens to indices, truncate/pad to max_len."""
    ids = [word2idx.get(t, word2idx[UNK_TOKEN]) for t in tokens]
    ids = ids[:max_len]
    length = len(ids)
    ids += [word2idx[PAD_TOKEN]] * (max_len - length)
    return ids, length


def find_question_positions(q_tokens, a_tokens):
    """
    Find positions in the answer where question words appear.
    Returns a list of answer-position indices.
    This is the set Q_a from the paper: positions in the answer
    where a question word is found.
    """
    q_set = set(q_tokens)
    return [i for i, tok in enumerate(a_tokens) if tok in q_set]


class QADataset(Dataset):
    def __init__(self, hf_split, word2idx, qid_field='question_id'):
        self.samples = []
        for idx, ex in enumerate(hf_split):
            q_tok = tokenize(ex['question'])
            a_tok = tokenize(ex['answer'])
            q_ids, q_len = encode_and_pad(q_tok, MAX_Q_LEN, word2idx)
            a_ids, a_len = encode_and_pad(a_tok, MAX_A_LEN, word2idx)
            
            # positions of question words in the (truncated) answer
            q_positions = find_question_positions(q_tok, a_tok[:MAX_A_LEN])
            
            # Get question ID — use field if available, else hash the question text
            if qid_field in ex:
                qid = ex[qid_field]
            else:
                qid = hash(ex['question']) % (10**8)
            
            label = float(ex['label'])

            self.samples.append({
                'q_ids':  torch.tensor(q_ids,  dtype=torch.long),
                'a_ids':  torch.tensor(a_ids,  dtype=torch.long),
                'q_len':  q_len,
                'a_len':  a_len,
                'q_pos':  q_positions,
                'label':  torch.tensor(label,  dtype=torch.float32),
                'qid':    qid,
            })

    def __len__(self): 
        return len(self.samples)
    
    def __getitem__(self, idx): 
        return self.samples[idx]


def collate_fn(batch):
    """Custom collate: stack tensors, keep q_pos as list-of-lists."""
    return {
        'q_ids':  torch.stack([b['q_ids'] for b in batch]),
        'a_ids':  torch.stack([b['a_ids'] for b in batch]),
        'q_len':  torch.tensor([b['q_len'] for b in batch]),
        'a_len':  torch.tensor([b['a_len'] for b in batch]),
        'q_pos':  [b['q_pos'] for b in batch],
        'label':  torch.stack([b['label'] for b in batch]),
        'qid':    [b['qid'] for b in batch],
    }
