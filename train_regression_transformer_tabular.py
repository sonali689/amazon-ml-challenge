#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Transformer + Tabular regression fine-tuning for ML Challenge 2025 - Smart Product Pricing

This script:
- Works on the original "normal" dataset (text-only) *and* on a feature-engineered dataset with extra columns.
- Auto-detects usable numeric tabular features and avoids target leakage by default (excludes columns matching
  'price', 'price_per', 'log_price', or exactly 'y'). You can override with CLI flags.
- Concatenates pooled Transformer text embeddings with standardized tabular features, then predicts log(price+1).
- Keeps a similar CLI to the original, plus a few new flags to control tabular features.
- Outputs the same `test_out.csv` with strictly positive prices.

Notes
-----
- Uses log1p(target) during training; predictions are expm1(logits), clipped to be positive.
- Computes SMAPE/MAE/RMSE on original scale.
- Saves per-fold scaler stats and selected tabular columns in `output_dir/fold*/tabular_config.json`.
"""

import os
import re
import math
import json
import argparse
import inspect
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import KFold, train_test_split
from transformers import AutoTokenizer, AutoModel
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments, set_seed
from transformers.trainer_callback import EarlyStoppingCallback
from transformers.trainer_utils import EvalPrediction


# ---------------------------
# Utilities
# ---------------------------

def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    SMAPE in [0, 200]
    SMAPE = (1/n) * sum( |y - yhat| / ((|y| + |yhat|)/2 + eps) ) * 100
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    num = np.abs(y_true - y_pred)
    den = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + eps
    return float(np.mean(num / den) * 100.0)


def positive_clip(x: np.ndarray, min_value: float = 1e-6) -> np.ndarray:
    return np.maximum(x, min_value)


def find_tabular_columns(
    df: pd.DataFrame,
    id_col: str,
    label_col: str,
    text_col: str,
    exclude_regex: str
) -> List[str]:
    """
    Auto-select numeric/boolean columns to be used as tabular features,
    excluding id/label/text columns and anything matching exclude_regex (case-insensitive).
    """
    pattern = re.compile(exclude_regex, flags=re.IGNORECASE)
    cols = []
    for c in df.columns:
        if c in (id_col, label_col, text_col):
            continue
        if pattern.search(c):
            continue
        dt = df[c].dtype
        if pd.api.types.is_bool_dtype(dt) or pd.api.types.is_numeric_dtype(dt):
            cols.append(c)
    return cols


def fit_scaler(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    means = np.nanmean(X, axis=0)
    stds = np.nanstd(X, axis=0)
    stds = np.where(stds < 1e-6, 1.0, stds)
    return means, stds


def apply_scaler(X: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    return (X - means) / stds


def save_tabular_config(path: str, tab_cols: List[str], means: np.ndarray, stds: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cfg = {"tabular_columns": tab_cols, "means": means.tolist(), "stds": stds.tolist()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def load_tabular_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------
# Dataset (tokenizer + tabular features)
# ---------------------------

class TextTabularRegDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        text_col: str,
        label_col: Optional[str],
        max_length: int,
        train_on_log: bool,
        tabular_cols: Optional[List[str]] = None,
        tab_means: Optional[np.ndarray] = None,
        tab_stds: Optional[np.ndarray] = None
    ):
        self.df = df.reset_index(drop=True).copy()
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.label_col = label_col
        self.max_length = max_length
        self.train_on_log = train_on_log
        self.tabular_cols = tabular_cols or []
        # Prepare tabular matrix (even if empty)
        if self.tabular_cols:
            X = self.df[self.tabular_cols].astype(float).values
            X = np.where(np.isfinite(X), X, np.nan)
            col_means = np.nanmean(X, axis=0)
            X = np.where(np.isnan(X), col_means, X)
            if tab_means is not None and tab_stds is not None:
                X = apply_scaler(X, tab_means, tab_stds)
            self.X_tab = torch.tensor(X, dtype=torch.float32)
        else:
            self.X_tab = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        text = str(row[self.text_col]) if not pd.isna(row[self.text_col]) else ""
        # IMPORTANT: don't request token_type_ids (prevents DistilBERT crash)
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_token_type_ids=False
        )
        if self.label_col is not None and self.label_col in row:
            price = float(row[self.label_col])
            label = math.log1p(price) if self.train_on_log else price
            enc["labels"] = torch.tensor(label, dtype=torch.float32)
        if self.X_tab is not None:
            enc["tab_features"] = self.X_tab[idx]
        return enc


# ---------------------------
# Custom collator (handles labels & tabular)
# ---------------------------

class TabularCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        tab_present = "tab_features" in features[0]
        tab_batch = None
        if tab_present:
            tab_batch = torch.stack([f["tab_features"] for f in features], dim=0)
            features = [{k: v for k, v in f.items() if k != "tab_features"} for f in features]

        # Convert scalar labels to python floats so tokenizer.pad can collate
        if "labels" in features[0]:
            for f in features:
                if isinstance(f.get("labels"), torch.Tensor) and f["labels"].ndim == 0:
                    f["labels"] = float(f["labels"].item())

        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")

        # Ensure labels are a float tensor
        if "labels" in batch and not torch.is_tensor(batch["labels"]):
            batch["labels"] = torch.tensor(batch["labels"], dtype=torch.float32)

        if tab_present:
            batch["tab_features"] = tab_batch
        return batch


# ---------------------------
# Model (text encoder + tabular head)
# ---------------------------

class TextTabularRegressor(nn.Module):
    def __init__(
        self,
        base_model_name: str,
        n_tab_features: int = 0,
        dropout: float = 0.1,
        hidden_mult: float = 0.5
    ):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(base_model_name)
        self.accepts_token_type_ids = "token_type_ids" in inspect.signature(self.text_model.forward).parameters

        hidden = self.text_model.config.hidden_size
        self.n_tab = int(n_tab_features)
        in_dim = hidden + (self.n_tab if self.n_tab > 0 else 0)
        mid_dim = max(16, int(hidden * hidden_mult))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, 1)
        )
        self.loss_fn = nn.MSELoss()

    @staticmethod
    def _mean_pool(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        return summed / counts

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        tab_features=None,
        labels=None
    ):
        # Build kwargs only with what the backbone accepts
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask, "return_dict": True}
        if token_type_ids is not None and self.accepts_token_type_ids:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.text_model(**kwargs)
        pooled = self._mean_pool(outputs.last_hidden_state, attention_mask)

        x = torch.cat([pooled, tab_features], dim=1) if tab_features is not None else pooled
        x = self.drop(x)
        pred_log = self.head(x).squeeze(-1)  # (B,)

        out = {"logits": pred_log.unsqueeze(-1)}
        if labels is not None:
            loss = self.loss_fn(pred_log, labels)
            out["loss"] = loss
        return out


# ---------------------------
# Metrics wrapper
# ---------------------------

@dataclass
class MetricsContext:
    y_true_log: np.ndarray  # log1p(y_true) if train_on_log=True else y_true
    train_on_log: bool = True

    def compute(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        logits = eval_pred.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        preds_log = np.array(logits).reshape(-1)

        if self.train_on_log:
            preds = np.expm1(preds_log)
            y_true = np.expm1(self.y_true_log)
        else:
            preds = preds_log
            y_true = self.y_true_log

        preds = positive_clip(preds, 1e-6)

        sm = smape(y_true, preds)
        mae = float(np.mean(np.abs(y_true - preds)))
        rmse = float(np.sqrt(np.mean((y_true - preds) ** 2)))
        return {"smape": sm, "mae": mae, "rmse": rmse}


# ---------------------------
# Training + Inference
# ---------------------------

def fit_single_fold(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    model_name: str,
    text_col: str,
    label_col: str,
    id_col: str,
    max_length: int,
    batch_size: int,
    lr: float,
    epochs: int,
    weight_decay: float,
    warmup_ratio: float,
    seed: int,
    output_dir: str,
    fp16: bool,
    log_every_steps: int,
    patience: int,
    gradient_accumulation_steps: int,
    tabular_cols: Optional[List[str]] = None
) -> Dict[str, Any]:
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Fit scaler on training tabular data
    tabular_cols = tabular_cols or []
    if tabular_cols:
        X_tr = train_df[tabular_cols].astype(float).values
        X_tr = np.where(np.isfinite(X_tr), X_tr, np.nan)
        col_means = np.nanmean(X_tr, axis=0)
        X_tr = np.where(np.isnan(X_tr), col_means, X_tr)
        means, stds = fit_scaler(X_tr)
        save_tabular_config(os.path.join(output_dir, "tabular_config.json"), tabular_cols, means, stds)
    else:
        means = stds = None

    # Datasets
    train_ds = TextTabularRegDataset(
        train_df, tokenizer, text_col, label_col, max_length, train_on_log=True,
        tabular_cols=tabular_cols, tab_means=means, tab_stds=stds
    )
    y_true_log = np.log1p(valid_df[label_col].astype(float).values)
    valid_ds = TextTabularRegDataset(
        valid_df, tokenizer, text_col, label_col, max_length, train_on_log=True,
        tabular_cols=tabular_cols, tab_means=means, tab_stds=stds
    )

    # Model + collator
    n_tab = len(tabular_cols)
    model = TextTabularRegressor(model_name, n_tab_features=n_tab)
    collator = TabularCollator(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_steps=log_every_steps,
        eval_strategy="steps",
        eval_steps=log_every_steps,
        save_strategy="steps",
        save_steps=log_every_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="smape",
        greater_is_better=False,
        fp16=fp16,
        gradient_accumulation_steps=gradient_accumulation_steps,
        report_to="none"
    )

    metrics_ctx = MetricsContext(y_true_log=y_true_log, train_on_log=True)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=tokenizer,  # OK (deprecation warning only)
        data_collator=collator,
        compute_metrics=metrics_ctx.compute,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)]
    )

    trainer.train()
    best_metrics = trainer.evaluate()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "metrics": best_metrics,
        "tabular_cols": tabular_cols,
        "tabular_means": means,
        "tabular_stds": stds
    }


def predict_prices(
    model: nn.Module,
    tokenizer,
    df: pd.DataFrame,
    text_col: str,
    max_length: int,
    batch_size: int,
    tabular_cols: Optional[List[str]] = None,
    tab_means: Optional[np.ndarray] = None,
    tab_stds: Optional[np.ndarray] = None
) -> np.ndarray:
    ds = TextTabularRegDataset(
        df, tokenizer, text_col=text_col, label_col=None,
        max_length=max_length, train_on_log=True,
        tabular_cols=tabular_cols, tab_means=tab_means, tab_stds=tab_stds
    )
    collator = TabularCollator(tokenizer=tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    preds = []
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=collator
    )
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs["logits"].squeeze(-1).detach().cpu().numpy()
            preds.append(logits)
    preds_log = np.concatenate(preds, axis=0)
    preds_price = np.expm1(preds_log)
    preds_price = positive_clip(preds_price, 1e-6)
    return preds_price


def kfold_fit_and_predict(
    train_csv: str,
    test_csv: str,
    text_col: str,
    label_col: str,
    id_col: str,
    model_name: str,
    output_csv: str,
    output_dir: str,
    folds: int,
    valid_size: float,
    random_state: int,
    max_length: int,
    batch_size: int,
    lr: float,
    epochs: int,
    weight_decay: float,
    warmup_ratio: float,
    patience: int,
    fp16: bool,
    gradient_accumulation_steps: int,
    log_every_steps: int,
    tabular_mode: str,
    tabular_cols_file: str,
    tabular_exclude_regex: str
):
    # Load data
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    for df in (train_df, test_df):
        if text_col not in df.columns:
            raise ValueError(f"Missing text column '{text_col}' in CSV.")
        df[text_col] = df[text_col].fillna("")

    # Determine tabular columns
    tab_cols: List[str] = []
    if tabular_mode == "none":
        tab_cols = []
    elif tabular_mode == "file":
        if not tabular_cols_file or not os.path.exists(tabular_cols_file):
            raise FileNotFoundError("--tabular_cols_file must be a valid path when --tabular_mode=file")
        with open(tabular_cols_file, "r", encoding="utf-8") as f:
            tab_cols = [line.strip() for line in f if line.strip()]
    else:  # auto
        tab_cols = find_tabular_columns(train_df, id_col=id_col, label_col=label_col, text_col=text_col,
                                        exclude_regex=tabular_exclude_regex)

    # Verify that chosen tabular columns exist in both train and test
    tab_cols = [c for c in tab_cols if (c in train_df.columns and c in test_df.columns)]
    print(f"[Tabular mode: {tabular_mode}] Using {len(tab_cols)} columns: {tab_cols[:8]}{'...' if len(tab_cols)>8 else ''}")

    models = []
    tokenizers = []
    val_scores = []
    fold_tabular_stats = []  # (tab_cols, means, stds)

    if folds <= 1:
        tr_df, va_df = train_test_split(train_df, test_size=valid_size, random_state=random_state)
        fold_dir = os.path.join(output_dir, "fold0")
        os.makedirs(fold_dir, exist_ok=True)
        artifacts = fit_single_fold(
            tr_df, va_df, model_name, text_col, label_col, id_col,
            max_length, batch_size, lr, epochs, weight_decay, warmup_ratio,
            random_state, fold_dir, fp16, log_every_steps, patience,
            gradient_accumulation_steps=gradient_accumulation_steps,
            tabular_cols=tab_cols
        )
        models.append(artifacts["model"])
        tokenizers.append(artifacts["tokenizer"])
        val_scores.append(artifacts["metrics"]["eval_smape"])
        fold_tabular_stats.append((tab_cols, artifacts["tabular_means"], artifacts["tabular_stds"]))
    else:
        kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
        for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(train_df)):
            tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
            va_df = train_df.iloc[va_idx].reset_index(drop=True)
            fold_dir = os.path.join(output_dir, f"fold{fold_idx}")
            os.makedirs(fold_dir, exist_ok=True)
            artifacts = fit_single_fold(
                tr_df, va_df, model_name, text_col, label_col, id_col,
                max_length, batch_size, lr, epochs, weight_decay, warmup_ratio,
                random_state + fold_idx, fold_dir, fp16, log_every_steps, patience,
                gradient_accumulation_steps=gradient_accumulation_steps,
                tabular_cols=tab_cols
            )
            models.append(artifacts["model"])
            tokenizers.append(artifacts["tokenizer"])
            val_scores.append(artifacts["metrics"]["eval_smape"])
            fold_tabular_stats.append((tab_cols, artifacts["tabular_means"], artifacts["tabular_stds"]))

    if len(val_scores) > 0:
        print(f"[Validation SMAPE per fold] {val_scores} | mean={np.mean(val_scores):.4f}")

    # Inference on test
    fold_preds = []
    for (m, tok), (tc, means, stds) in zip(zip(models, tokenizers), fold_tabular_stats):
        fold_preds.append(
            predict_prices(m, tok, test_df, text_col=text_col,
                           max_length=max_length, batch_size=batch_size,
                           tabular_cols=tc, tab_means=means, tab_stds=stds)
        )
    test_pred = np.mean(np.stack(fold_preds, axis=0), axis=0)

    # Output
    if id_col not in test_df.columns:
        raise ValueError(f"Missing id column '{id_col}' in test CSV.")
    out = pd.DataFrame({
        id_col: test_df[id_col].values,
        "price": positive_clip(test_pred.astype(float), 1e-6)
    })
    out.to_csv(output_csv, index=False)
    print(f"Wrote predictions to: {output_csv}")


# ---------------------------
# CLI
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Transformer(+Tabular) regression fine-tuning")
    p.add_argument("--train_csv", type=str, default="dataset/train.csv", help="Path to training CSV with price column")
    p.add_argument("--test_csv", type=str, default="dataset/test.csv", help="Path to test CSV without price")
    p.add_argument("--text_col", type=str, default="catalog_content")
    p.add_argument("--label_col", type=str, default="price")
    p.add_argument("--id_col", type=str, default="sample_id")
    p.add_argument("--model_name", type=str, default="distilbert-base-uncased",
                   help="HF model name (e.g., distilbert-base-uncased, microsoft/deberta-v3-base)")
    p.add_argument("--output_csv", type=str, default="test_out.csv")
    p.add_argument("--output_dir", type=str, default="checkpoints")
    p.add_argument("--folds", type=int, default=1, help="K-fold splits (1 = no CV)")
    p.add_argument("--valid_size", type=float, default=0.1)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--log_every_steps", type=int, default=200)

    # Tabular feature control
    p.add_argument("--tabular_mode", type=str, default="auto", choices=["auto", "none", "file"],
                   help="auto-detect, none (text-only), or read from file")
    p.add_argument("--tabular_cols_file", type=str, default="",
                   help="Path to a text file with one column name per line (used when --tabular_mode=file)")
    p.add_argument("--tabular_exclude_regex", type=str, default=r"(?:^y$|price_per|price|log_price)",
                   help="Regex of columns to exclude (to avoid leakage), case-insensitive")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    kfold_fit_and_predict(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        text_col=args.text_col,
        label_col=args.label_col,
        id_col=args.id_col,
        model_name=args.model_name,
        output_csv=args.output_csv,
        output_dir=args.output_dir,
        folds=args.folds,
        valid_size=args.valid_size,
        random_state=args.random_state,
        max_length=args.max_length,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        patience=args.patience,
        fp16=args.fp16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_every_steps=args.log_every_steps,
        tabular_mode=args.tabular_mode,
        tabular_cols_file=args.tabular_cols_file,
        tabular_exclude_regex=args.tabular_exclude_regex
    )


if __name__ == "__main__":
    main()
