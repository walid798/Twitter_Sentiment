# src/train.py
import os
import math
import json
import random
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report

from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.data.datasets import TweetDataset
from src.data.preprocess import load_vocab


# --------------------------
# Utilities
# --------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_class_weights(labels, num_classes=3):
    """
    Inverse frequency class weights for CrossEntropyLoss.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (counts * num_classes)
    return torch.tensor(weights, dtype=torch.float32)


@dataclass
class TrainConfig:
    # data
    train_csv: str
    val_csv: str
    text_col: str = "text"
    label_col: str = "label"
    max_len: int = 50
    clean: bool = True

    # model
    model_type: str = "lstm"  # or "bert"
    vocab_path: Optional[str] = None      # for LSTM
    pretrained_name: str = "bert-base-uncased"  # for BERT

    # training
    batch_size: int = 64
    epochs: int = 10
    lr: float = 1e-3
    warmup_ratio: float = 0.1  # BERT only
    class_weights: bool = True
    patience: int = 3
    mixed_precision: bool = True  # use amp for BERT

    # misc
    seed: int = 42
    save_dir: str = "experiments/runs/lstm"  # or .../bert
    label_map: Optional[Dict[str, int]] = None

    


# --------------------------
# DataLoaders
# --------------------------
def build_dataloaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    if cfg.label_map is None:
        label_map = {"negative": 0, "neutral": 1, "positive": 2}
    else:
        label_map = cfg.label_map

    if cfg.model_type == "bert":
        tok = AutoTokenizer.from_pretrained(cfg.pretrained_name)
        train_ds = TweetDataset(
            csv_path=cfg.train_csv,
            text_col=cfg.text_col,
            label_col=cfg.label_col,
            tokenizer=tok,
            max_len=cfg.max_len,
            label_map=label_map,
            clean=cfg.clean,
        )
        val_ds = TweetDataset(
            csv_path=cfg.val_csv,
            text_col=cfg.text_col,
            label_col=cfg.label_col,
            tokenizer=tok,
            max_len=cfg.max_len,
            label_map=label_map,
            clean=cfg.clean,
        )
    else:
        if not cfg.vocab_path:
            raise ValueError("For LSTM, cfg.vocab_path must be set.")
        vocab = load_vocab(cfg.vocab_path)
        train_ds = TweetDataset(
            csv_path=cfg.train_csv,
            text_col=cfg.text_col,
            label_col=cfg.label_col,
            vocab=vocab,
            max_len=cfg.max_len,
            label_map=label_map,
            clean=cfg.clean,
        )
        val_ds = TweetDataset(
            csv_path=cfg.val_csv,
            text_col=cfg.text_col,
            label_col=cfg.label_col,
            vocab=vocab,
            max_len=cfg.max_len,
            label_map=label_map,
            clean=cfg.clean,
        )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, label_map


# --------------------------
# Train & Validate
# --------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * input_ids.size(0)
        preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    return avg_loss, acc, f1_macro


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)

        total_loss += loss.item() * input_ids.size(0)
        preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, target_names=["NEG", "NEU", "POS"], digits=4)
    return avg_loss, acc, f1_macro, report


# --------------------------
# Orchestration
# --------------------------
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from src.models.bert import BertClassifier  
from src.models.lstm import LSTMClassifier
def fit(cfg: TrainConfig):
    seed_everything(cfg.seed)
    device = get_device()
    os.makedirs(cfg.save_dir, exist_ok=True)

    # Data
    train_loader, val_loader, label_map = build_dataloaders(cfg)

    # Model
    if cfg.model_type == "bert":
        print("BERT")
        # model = BertClassifier(model_name=cfg.pretrained_name, num_classes=3)
    else:
        
        vocab = load_vocab(cfg.vocab_path)
        model = LSTMClassifier(vocab_size=len(vocab))

    model.to(device)

    # Loss (optionally class-weighted)
    train_labels = np.array(train_loader.dataset.labels)
    if cfg.class_weights:
        weights = compute_class_weights(train_labels, num_classes=3).to(device)
    else:
        weights = None
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Optim + sched
    if cfg.model_type == "bert":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        total_steps = cfg.epochs * math.ceil(len(train_loader.dataset) / cfg.batch_size)
        warmup_steps = int(cfg.warmup_ratio * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.mixed_precision and device.type == "cuda")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        scheduler = None
        scaler = None

    # Early stopping
    best_f1 = -1.0
    best_path = os.path.join(cfg.save_dir, "best.pt")
    patience_cnt = 0

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        if scheduler is not None:
            scheduler.step()
        va_loss, va_acc, va_f1, va_report = validate(model, val_loader, criterion, device)

        print(f"[Epoch {epoch:02d}] "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} f1={tr_f1:.4f} | "
              f"val_loss={va_loss:.4f} acc={va_acc:.4f} f1={va_f1:.4f}")

        # Checkpoint
        if va_f1 > best_f1:
            best_f1 = va_f1
            patience_cnt = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": cfg.__dict__,
                    "label_map": label_map,
                },
                best_path,
            )
            print(f"  ✅ Saved new best to {best_path}")
        else:
            patience_cnt += 1
            if patience_cnt >= cfg.patience:
                print("  ⏹ Early stopping triggered.")
                break

    print("\nValidation classification report (best epoch may differ):")
    print(va_report)
    print(f"\nBest macro-F1: {best_f1:.4f}  |  Checkpoint: {best_path}")
    return best_path
