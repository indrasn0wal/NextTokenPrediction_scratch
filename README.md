# Neural Language Model (PyTorch) – Assignment 2

This repository contains the implementation of a neural language model trained from scratch using PyTorch.  
The model is a Transformer-based decoder trained on the provided text dataset using a custom Byte Pair Encoding (BPE) tokenizer.

---

## Project Overview

The goal of this assignment is to:

- Implement a language model from scratch (no pre-trained models)
- Build a tokenizer manually (BPE)
- Prepare the dataset and batching logic
- Train the model using teacher forcing
- Produce training and validation loss curves
- Evaluate the model using perplexity
- Demonstrate underfitting, overfitting, and best-fit configurations
- Ensure reproducibility

---

## Repository Structure

```
├── train_model.ipynb        # Main training notebook
├── transformer_overfit.pth  # Saved model weights (example)
├── bpe_tokenizer.pkl        # Serialized BPE tokenizer
├── bpe_vocab.json           # Vocabulary file
├── README.md                # This file
```

---

## Dataset

- Dataset: Provided text file (e.g., Pride and Prejudice, public domain)
- Tokenization: Custom Byte Pair Encoding (BPE)
- Final vocab size: 880 tokens
- Train/validation split: 90% / 10%
- Total encoded tokens: 334,616

---

## Methodology

### 1. Tokenization
- Implemented a full BPE tokenizer from scratch.
- Starts from characters and merges frequent pairs.
- Includes 4 special tokens: `<PAD>`, `<UNK>`, `<SOS>`, `<EOS>`.

### 2. Data Processing
- Lines shuffled to avoid chapter-wise ordering bias.
- Encoded using the trained BPE tokenizer.
- Sequences generated using sliding windows:
  - Sequence length: 128
  - Batch size: 64

### 3. Model Architecture
A custom Transformer decoder with:
- Token + positional embeddings
- Masked multi-head self-attention
- GELU feed-forward layers
- LayerNorm and residual connections
- Linear output projection

### 4. Training
- Optimizer: Adam  
- Learning rate: 2e-4  
- Loss: Cross entropy  
- Metric: Perplexity  
- Epochs: 10  

### 5. Reproducibility
Seeds set for:
- Python random
- NumPy
- PyTorch (CPU and CUDA)
- cuDNN made deterministic

---

## Experiments

### Underfitting
- Very small model
- Heavy dropout (0.5)
- Result: High train/validation loss

### Overfitting
- Large model
- Dropout = 0.0
- Result: Train loss decreased, validation loss increased

### Best Fit
- Medium-sized transformer
- Dropout = 0.1–0.3
- Both losses decreased smoothly
- Best perplexity

---

## Outputs

The notebook produces:
- Training curves  
- Validation curves  
- Perplexity for all runs  
- Saved model weights  
- Saved tokenizer and vocabulary  

---

## How to Run

### 1. Clone the repository
```bash
git clone <repo_url>
cd <repo_name>
```

### 2. Install dependencies
```bash
pip install torch numpy matplotlib
```

### 3. Run training
Open the notebook in colab notebook, and upload the .txt file in colab, and run line by line:
```
train_model.ipynb
```

---

## Notes
- No pre-trained models or high-level LM libraries are used.
- All components are built manually using PyTorch.
- Only the provided dataset is used for training.
