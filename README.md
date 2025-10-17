## 🤖 Sentiment Classification on IMDB using Transformer Fine-Tuning

This repository contains a Natural Language Processing (NLP) project that fine-tunes pretrained Transformer models to classify movie reviews from the IMDB dataset as **positive** or **negative**.  
Two fine-tuning strategies are implemented:
1. **Full Fine-Tuning** – all model parameters are updated.  
2. **LoRA Fine-Tuning** – only low-rank adapter weights are trained for efficiency.  

The project compares both methods in terms of training loss, convergence behavior, and computational efficiency.

---


---

## ⚙️ Project Workflow

### 1. Data Preparation
- The **IMDB dataset** is automatically loaded using Hugging Face Datasets (`load_dataset("imdb")`).  
- Texts are tokenized with the **DistilBERT tokenizer** (`max_length=256`, truncation + padding).  
- Data is split into training and test sets.

### 2. Model Fine-Tuning
- **Full Fine-Tuning:** all model weights updated using AdamW optimizer and linear LR scheduler.  
- **LoRA Fine-Tuning:** adds low-rank adapters to transformer layers; only adapter weights are trainable.

### 3. Evaluation
- Accuracy and macro F1 are computed on the test split.  
- Training loss is plotted to compare learning dynamics.

### 4. Visualization
- The code includes plot such as:
  - `Training Loss vs Steps` (for both setups)

---

## 🧮 Experimental Setup

| Setting | Value |
|----------|--------|
| Model | `distilbert-base-uncased` |
| Dataset | IMDB (25k train / 25k test) |
| Task | Binary sentiment classification |
| Batch Size | 32 |
| Learning Rate | 2e-5 for full fine-tuning, 2e-4 for LoRA |
| Epochs | 2 |
| Weight Decay | 0.01 |
| Evaluation Metric | Accuracy, F1-macro |

---

## 📊 Results Overview

| Model | Final Training Loss | Accuracy | F1-score | Observations |
|--------|----------------------|-----------|-----------|---------------|
| Full Fine-Tuning | ~0.15 | ≈ 93 % | ≈ 0.92 | Smooth convergence, slightly lower final loss |
| LoRA | ~0.12 – 0.17 (stable) | ≈ 92 % | ≈ 0.91 | Similar performance, higher efficiency |

---

## 🧱 Reproducibility

To reproduce training:

```bash
# run main.py


