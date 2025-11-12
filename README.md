# Amazon ML Challenge — Smart Product Pricing

**One-line:** Reproducible pipeline that fine-tunes a Transformer on product text, fuses optional tabular features, and predicts strictly-positive product prices for the ML Challenge 2025.

---

## Overview
This repository implements a leakage-aware, production-ready pipeline to predict product prices from `catalog_content` (title/description/IPQ) and optional numeric/tabular signals. The model is trained in log-space for numerical stability and evaluated with **SMAPE** (competition metric).

---

## Key features
- Text backbone: HuggingFace Transformer (e.g. `distilbert-base-uncased` / `microsoft/deberta-v3-base`).  
- Tabular auto-discovery: automatically selects numeric/boolean columns while excluding ID/text/price/leakage-prone fields.  
- Fusion head: mean-pooled text embeddings concatenated with standardized tabular features → 2-layer MLP head.  
- Training on `log(1 + price)` (MSE); inference applies `exp(z) - 1` and clips to positive floats.  
- K-Fold CV with fold checkpoint ensembling and deterministic preprocessing per fold.  
- Designed to produce portal-ready `test_out.csv` with exact columns `sample_id,price`.

---

## Metric
- **SMAPE** (Symmetric Mean Absolute Percentage Error) is used for evaluation:
  
  \[
  \text{SMAPE} = \frac{1}{n}\sum_{i=1}^n \frac{|\hat y_i - y_i|}{\tfrac{1}{2}(|\hat y_i| + |y_i|) + \varepsilon}
  \]

---
## Data
- `dataset/train.csv` — training rows (contains `price`).  
- `dataset/test.csv` — test rows (no `price`).

## Quick start (examples)

```bash
# quick run (single split)
python train_regression_transformer_tabular.py \
  --train_csv dataset/train.csv \
  --test_csv dataset/test.csv \
  --output_csv test_out.csv \
  --output_dir checkpoints \
  --model_name distilbert-base-uncased \
  --folds 1 --valid_size 0.1 --fp16

# 5-fold CV with auto tabular discovery
python train_regression_transformer_tabular.py \
  --train_csv dataset/train.csv \
  --test_csv dataset/test.csv \
  --output_csv test_out.csv \
  --output_dir checkpoints \
  --model_name microsoft/deberta-v3-base \
  --folds 5 --patience 5 --log_every_steps 200 --fp16

