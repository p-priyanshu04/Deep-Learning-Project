# Reproducing: Enhancing Recurrent Neural Networks with Positional Attention for Question Answering

> Chen et al., SIGIR 2017 — [[Paper]](https://doi.org/10.1145/3077136.3080699)

This repository reproduces the **RNN-POA** (Positional Attention) model for answer selection, evaluated on WikiQA and TREC-QA (clean) datasets.

---

## Overview

Standard attention mechanisms in RNN-based QA models rely purely on hidden representations, ignoring where question words appear in the answer. This paper proposes **positional attention** — if a question word occurs in an answer, neighbouring words should receive more attention via a Gaussian-kernel influence propagation.

**Key contributions reproduced:**
- Shared Bidirectional LSTM (BLSTM) encoder for questions and answers
- Gaussian-kernel position-aware influence propagation
- Positional attention mechanism over answer hidden states
- Manhattan distance similarity with ℓ₁ norm

---

## Model Architecture

```
Question ──► BLSTM ──► Classical Attention ──► r_q ──┐
                                                       ├──► exp(-||r_q - r_a||_1) ──► similarity
Answer  ──► BLSTM ──► Positional Attention ──► r_a ──┘
                            ▲
               Gaussian Kernel Influence (d̂)
               computed from question-word positions
```

Three models are implemented and compared:

| Model | Question Repr. | Answer Repr. | Position-aware |
|-------|---------------|--------------|----------------|
| RNN-AVG | mean pool | mean pool | ✗ |
| RNN-ATT | mean pool | classical attention | ✗ |
| RNN-POA | classical attention | positional attention | ✓ |

---

## Results

### WikiQA

| Model | MAP | MRR |
|-------|-----|-----|
| RNN-AVG | 0.6889 | 0.6999 |
| RNN-ATT | 0.6961 | 0.7085 |
| RNN-POA (paper) | **0.7212** | **0.7312** |
| RNN-POA (ours) | - | - |

### TREC-QA (Clean)

| Model | MAP | MRR |
|-------|-----|-----|
| RNN-AVG | 0.7064 | 0.8086 |
| RNN-ATT | 0.7180 | 0.8121 |
| RNN-POA (paper) | **0.7814** | **0.8513** |
| RNN-POA (ours) | - | - |



---

## Setup

### Requirements

```bash
pip install git+https://github.com/p-priyanshu04/Deep-Learning-Project.git
```

GloVe embeddings (100d) are downloaded automatically from Stanford NLP on first run.

### Datasets

Both datasets are loaded via HuggingFace `datasets`:

```python
wiki_qa     = load_dataset('wiki_qa')
trec_qa_raw = load_dataset('lucadiliello/trecqa')
```

---

## Repository Structure

```

```

---

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `HIDDEN_DIM` | 50 | BLSTM hidden size per direction (2×50=100 total) |
| `EMBED_DIM` | 100 | GloVe embedding dimension |
| `SIGMA_SCOPE` (σ) | tuned {5..55} | Gaussian influence propagation scope |
| `SIGMA_PRIME` (σ') | 0.1 | Gaussian std dev for influence base matrix K |
| `BATCH_SIZE` | 64 | Training batch size |
| `N_EPOCHS` | 30 | Max training epochs |
| `PATIENCE` | 5 | Early stopping patience (based on dev MAP) |
| `LR` | 1.0 | Adadelta learning rate |
| `MAX_Q_LEN` | 40 | Max question length (tokens) |
| `MAX_A_LEN` | 100 | Max answer length (tokens) |

The propagation scope σ is tuned over {5, 15, 25, 35, 45, 55} on the WikiQA dev set, with the best value selected by MAP.

---

## Training

The notebook runs the following pipeline:

1. Download and load GloVe 100d embeddings
2. Load WikiQA and TREC-QA from HuggingFace
3. Build shared vocabulary and embedding matrix
4. Tune σ on WikiQA dev set
5. Train RNN-AVG, RNN-ATT, RNN-POA on both datasets
6. Evaluate on test sets with MAP and MRR
7. Visualize training curves, attention heatmaps, and position influence

---

## Visualizations

The notebook produces the following plots:

- `training_curves.png` — Train/dev loss, MAP, MRR per epoch for both datasets
- `dataset_comparison.png` — Bar chart comparing MAP/MRR across WikiQA and TREC-QA
- `paper_vs_ours.png` — Our results vs paper reported results
- `sigma_tuning.png` — Effect of σ on dev MAP and MRR
- `attention_heatmap_example_N.png` — RNN-ATT vs RNN-POA attention weights side by side
- `position_influence_example_N.png` — Gaussian influence propagation (d̂) visualization

---

## Citation

```bibtex
@inproceedings{chen2017enhancing,
  title     = {Enhancing Recurrent Neural Networks with Positional Attention for Question Answering},
  author    = {Chen, Qin and Hu, Qinmin and Huang, Jimmy Xiangji and He, Liang and An, Weijie},
  booktitle = {Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages     = {993--996},
  year      = {2017}
}
```