# src/evaluate.py
import os
import json
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

from src.data.datasets import TweetDataset
from src.data.preprocess import load_vocab


LABELS = ["NEG", "NEU", "POS"]  # 0,1,2


def _to_numpy_logits_probs(logits: torch.Tensor) -> np.ndarray:
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    return probs


@torch.no_grad()
def predict_model(model, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_prob) where y_prob is [N, 3]."""
    model.eval()
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].cpu().numpy()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = _to_numpy_logits_probs(logits)
        all_probs.append(probs)
        all_labels.append(labels)

    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    return y_true, y_prob


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
    y_pred = y_prob.argmax(axis=1)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0,1,2], zero_division=0)
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, target_names=LABELS, digits=4)
    # OvR ROC-AUC (requires probabilities)
    try:
        y_true_ovr = np.eye(3)[y_true]
        roc_auc_ovr = roc_auc_score(y_true_ovr, y_prob, average="macro", multi_class="ovr")
    except Exception:
        roc_auc_ovr = float("nan")
    return {
        "accuracy": acc,
        "macro": {"precision": macro_prec, "recall": macro_rec, "f1": macro_f1},
        "per_class": {
            "NEG": {"precision": prec[0], "recall": rec[0], "f1": f1[0], "support": int(support[0])},
            "NEU": {"precision": prec[1], "recall": rec[1], "f1": f1[1], "support": int(support[1])},
            "POS": {"precision": prec[2], "recall": rec[2], "f1": f1[2], "support": int(support[2])},
        },
        "roc_auc_ovr": roc_auc_ovr,
        "report": report,
    }


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, out_path: Optional[str] = None):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=LABELS,
        yticklabels=LABELS,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    # annotate
    thresh = cm.max() / 2.0 if cm.size else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_roc_ovr(y_true: np.ndarray, y_prob: np.ndarray, out_path: Optional[str] = None):
    # One-vs-rest ROC for 3 classes
    y_true_ovr = np.eye(3)[y_true]
    fig, ax = plt.subplots()
    for i, name in enumerate(LABELS):
        fpr, tpr, _ = roc_curve(y_true_ovr[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, label=f"{name}")
    ax.plot([0,1], [0,1], linestyle="--")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC (OvR)")
    ax.legend()
    if out_path:
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_pr_ovr(y_true: np.ndarray, y_prob: np.ndarray, out_path: Optional[str] = None):
    y_true_ovr = np.eye(3)[y_true]
    fig, ax = plt.subplots()
    for i, name in enumerate(LABELS):
        prec, rec, _ = precision_recall_curve(y_true_ovr[:, i], y_prob[:, i])
        ax.plot(rec, prec, label=f"{name}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall (OvR)")
    ax.legend()
    if out_path:
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _build_loader(split_csv: str, model_type: str, max_len: int, text_col: str, label_col: str,
                  vocab_path: Optional[str], pretrained_name: str, batch_size: int) -> DataLoader:
    if model_type == "bert":
        tok = AutoTokenizer.from_pretrained(pretrained_name)
        ds = TweetDataset(split_csv, text_col=text_col, label_col=label_col, tokenizer=tok, max_len=max_len)
    else:
        if not vocab_path:
            raise ValueError("LSTM eval requires vocab_path.")
        vocab = load_vocab(vocab_path)
        ds = TweetDataset(split_csv, text_col=text_col, label_col=label_col, vocab=vocab, max_len=max_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: str,
    split_csv: str,
    out_dir: str,
    model_type: str = "lstm",
    max_len: int = 50,
    text_col: str = "text",
    label_col: str = "label",
    vocab_path: Optional[str] = "experiments/runs/lstm/vocab.json",
    pretrained_name: str = "bert-base-uncased",
    batch_size: int = 128,
) -> Dict:
    """
    Load a saved checkpoint and evaluate on a given split CSV.
    Saves metrics.json and plots to out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Load model + config
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "bert":
        print("BERT")
        # from src.models.bert import BertClassifier  
        # model = BertClassifier(model_name=pretrained_name, num_classes=3)
    else:
        from src.models.lstm import LSTMClassifier
        vocab = load_vocab(vocab_path)
        model = LSTMClassifier(vocab_size=len(vocab))

    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    # Data
    loader = _build_loader(
        split_csv=split_csv,
        model_type=model_type,
        max_len=max_len,
        text_col=text_col,
        label_col=label_col,
        vocab_path=vocab_path,
        pretrained_name=pretrained_name,
        batch_size=batch_size,
    )

    # Predict
    y_true, y_prob = predict_model(model, loader, device)
    y_pred = y_prob.argmax(axis=1)

    # Metrics
    metrics = compute_metrics(y_true, y_prob)

    # Save metrics + plots
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": checkpoint_path,
                "split_csv": split_csv,
                **metrics,
            },
            f,
            indent=2,
        )

    plot_confusion(y_true, y_pred, os.path.join(out_dir, "confusion_matrix.png"))
    try:
        plot_roc_ovr(y_true, y_prob, os.path.join(out_dir, "roc_ovr.png"))
        plot_pr_ovr(y_true, y_prob, os.path.join(out_dir, "pr_ovr.png"))
    except Exception:
        pass

    print("✅ Evaluation complete. Saved to:", out_dir)
    print(metrics["report"])
    return metrics
