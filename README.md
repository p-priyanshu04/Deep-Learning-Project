# Enhancing Recurrent Neural Networks with Positional Attention for Question Answering

> **Reproduction of:** Chen et al., SIGIR 2017 [[Paper]](https://doi.org/10.1145/3077136.3080699)  
> This repository contains the implementation of the research paper on *Enhancing RNN with Positional Attention for Q&A*.

---

## Overview

Standard attention mechanisms in RNN-based QA models rely purely on hidden representations, ignoring where question words appear in the answer. This paper proposes **positional attention**—if question words are close to similar answer tokens, they should contribute more to the answer's encoding.

**Key contributions reproduced:**
- Shared Bidirectional LSTM (BLSTM) encoder for questions and answers
- Gaussian-kernel position-aware influence propagation
- Positional attention mechanism over answer hidden states
- Manhattan distance similarity with L1 norm

---

## Repository Structure

```
.
├── rnn_poa_reproduction.ipynb      # Main step-by-step reproduction notebook
├── rnn-poa.ipynb                   # Additional implementation/experiment notebook
├── FinalScripts/                   # Python scripts for each pipeline stage
│   ├── 1_data_preprocessing.py
│   ├── 2_baseline_metrics.py
│   ├── 3_model_implementation.py
│   ├── 4_train_evaluate.py
│   └── 5_results_visualize.py
├── notebooks/                      # Modular Jupyter workflow notebooks (mirrored from scripts)
│   ├── 01_Data_Preprocessing.ipynb
│   ├── 02_Model_Implementation.ipynb
│   ├── 03_Training_and_Experiments.ipynb
│   └── 04_Results_and_Analysis.ipynb
├── RNN-POA-Project/                # Archival or legacy modular notebook pipeline + code
│   ├── BaselineModel&EvalMetrics.ipynb
│   ├── DataPreprocessing (1).ipynb
│   ├── ModelImplementationipynb (1).ipynb
│   ├── ResultsAndVisualizations.ipynb
│   ├── TrainingAndEvaluation.ipynb
│   └── core_logic/                 # Auxiliary code (utils, models, etc.)
├── utils/                          # Utility Python modules
│   ├── data_utils.py               # Dataset loading and preprocessing helpers
│   ├── model_utils.py              # Model construction & manipulation
│   └── train_utils.py              # Training loop utilities
├── scratch/                        # Experimental code, prototypes, and analysis scripts
│   ├── models_implemented.py
│   └── ... (other experimental files)
├── patch_rnn_poa.py                # Patch or augmentation script
├── requirements.txt                # PIP requirements for Python dependencies
├── pyproject.toml                  # Project metadata/configuration
├── Enhancing Recurrent Neural Networks with Positional Attention for Question Answering.pdf   # Original paper
└── README.md
```

---

## How to Use

### Requirements

```bash
pip install -r requirements.txt
# Or, install the repo directly via pip
pip install git+https://github.com/p-priyanshu04/Deep-Learning-Project.git
```
GloVe embeddings (100d) are downloaded automatically from Stanford NLP on first run.

### Datasets

Datasets are loaded via HuggingFace `datasets`:
```python
from datasets import load_dataset
wiki_qa     = load_dataset('wiki_qa')
trec_qa_raw = load_dataset('lucadiliello/trecqa')
```

---

## Main Pipelines

The primary workflow for reproduction is in `rnn_poa_reproduction.ipynb` ([view here](https://github.com/p-priyanshu04/Deep-Learning-Project/blob/main/rnn_poa_reproduction.ipynb)).  
For modularized/scripted execution, use the sequential scripts in `FinalScripts/`:
1. `1_data_preprocessing.py` — Prepare data and vocabulary
2. `2_baseline_metrics.py` — Compute baseline model metrics
3. `3_model_implementation.py` — Defines RNN architectures (incl. POA)
4. `4_train_evaluate.py` — Handles training and evaluation routines
5. `5_results_visualize.py` — Visualization, plotting, and results analysis

You may also use the notebook versions under `/notebooks` for a step-wise interactive workflow.

---

## Model Architecture

```
Question ──► BLSTM ──► Classical Attention ──► r_q ──┐
                                                    ├─► exp(-||r_q - r_a||₁) ──► similarity
Answer   ──► BLSTM ──► Positional Attention ──► r_a ┘
                         ▲
          Gaussian Kernel Influence (d̂)
          computed from question-word positions
```

| Model    | Question Repr.     | Answer Repr.        | Position-aware |
|----------|--------------------|---------------------|---------------|
| RNN-AVG  | mean pool          | mean pool           | ✗             |
| RNN-ATT  | mean pool          | classical attention | ✗             |
| RNN-POA  | classical attention| positional attention| ✓             |

---

## Results

### WikiQA

| Model         | MAP    | MRR    |
|---------------|--------|--------|
| RNN-AVG       | 0.6889 | 0.6999 |
| RNN-ATT       | 0.6961 | 0.7085 |
| RNN-POA (paper) | **0.7212** | **0.7312** |
| RNN-POA (ours) | **0.7062**  | **0.7179**   |

### TREC-QA (Clean)

| Model             | MAP    | MRR    |
|-------------------|--------|--------|
| RNN-AVG           | 0.7064 | 0.8086 |
| RNN-ATT           | 0.7180 | 0.8121 |
| RNN-POA (paper)   | **0.7814** | **0.8513** |
| RNN-POA (ours)    | 0.5862     | 0.6890 |

---

## Visualizations

The pipeline generates:
- `training_curves.png` — Training/dev loss, MAP, MRR per epoch
- `dataset_comparison.png` — Bar chart comparing MAP/MRR across WikiQA and TREC-QA
- `paper_vs_ours.png` — Results comparison with reported paper values
- `sigma_tuning.png` — Effect of σ on dev MAP/MRR
- `attention_heatmap_example_N.png` — Illustrates difference between classical and positional attention
- `position_influence_example_N.png` — Visualization of Gaussian kernel position influence

---

## Reference

- Original research article: [`Enhancing Recurrent Neural Networks with Positional Attention for Question Answering`](Enhancing%20Recurrent%20Neural%20Networks%20with%20Positional%20Attention%20for%20Question%20Answering.pdf)
- Notebook reproduction: [`rnn_poa_reproduction.ipynb`](rnn_poa_reproduction.ipynb)
- Main scripts: [`FinalScripts/`](FinalScripts/)
- All notebook pipeline: [`notebooks/`](notebooks/)
- Reproducible environment: [`requirements.txt`](requirements.txt)

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
