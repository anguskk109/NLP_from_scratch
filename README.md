# BERT / GPT / T5 from Scratch

This repository implements three foundational transformer architectures **from the ground up** in pure PyTorch:

- **BERT-like**: Encoder-only model trained with **Masked Language Modeling (MLM)**
- **GPT-like**: Decoder-only model trained with **Causal Language Modeling (CLM)**
- **T5-like**: Encoder-decoder model trained with **Span Corruption**

The goal is **not to reproduce SOTA results**, but to **deeply understand and verify** the core structural and pretraining mechanics of modern NLP models through hands-on implementation.

---

## 🎯 Purpose & Philosophy

This project was created to **practice and internalize** the three major paradigms of transformer-based NLP:

1. **Bidirectional encoding** (BERT)
2. **Autoregressive generation** (GPT)
3. **Text-to-text denoising** (T5)

Every module (`attention.py`, `encoder_layer.py`, `prediction_head.py`, etc.) was built manually — **no model loading from `transformers`**, only tokenizer and dataset utilities were borrowed for practicality.

---

## 🔍 Implementation Notes

- ✅ **Architecturally serious**: All models use correct attention masking, pre-normalization, weight tying, and modular design.
- ⚠️ **Not 100% paper-faithful**: Some details (e.g., BERT’s embedding LayerNorm, T5’s relative positional bias) were simplified for clarity and unification.
- ⚠️ **Training is minimal but meaningful**:  
  - Dataset: `TinyStories` (small, fast, clean)  
  - Model size: Tiny (`hidden_size=256`, `num_layers=4`)  
  - Training: ~10,000–60,000 steps 
  - **Stopped early** once loss decreased and validation metrics became reasonable — **sufficient to verify correctness**, not to maximize performance.

---

## 🛠️ What This Project Does **Not** Include

This is **not a production training framework**. The following were intentionally omitted because they are **orthogonal to the core goal** (architecture practice):

- Learning rate warmup / scheduling
- Early stopping
- Model checkpointing (beyond final save)
- Downstream fine-tuning (e.g., GLUE, summarization)
- Advanced data augmentation or caching

> The focus is **model structure**, not training optimization.

---

## 📦 What’s Included

configs/ # Model hyperparameters (YAML)
models/ # BERT, GPT, T5 + shared modules
train/ # Pretraining scripts for all three
utils/ # Config, logging, helpers
requirements.txt # Minimal dependencies

Each model trains independently using:
- **BERT**: MLM only (no NSP)
- **GPT**: Causal LM with greedy generation
- **T5**: True span corruption (Bernoulli masking → contiguous spans)

All logs are saved locally in JSON format — **no external services required**.

---

## 🔮 Future Work

This project covers **upstream pretraining only**.  
A follow-up project will explore **downstream fine-tuning and probing** on tasks like classification, summarization, and question answering.

*(Link will be added here once the next project is published.)*

---

## 🙏 Acknowledgements

- Inspired by the original papers:  
  - [BERT (Devlin et al., 2018)](https://arxiv.org/abs/1810.04805)  
  - [GPT-2 (Radford et al., 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)  
  - [T5 (Raffel et al., 2020)](https://arxiv.org/abs/1910.10683)
- Tokenizers and datasets [TinyStories] from Hugging Face

---

> This repo is a **learning artifact** — built to understand, not to deploy.
