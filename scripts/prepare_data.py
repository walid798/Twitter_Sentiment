import os
import json
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.preprocess import clean_tweet, build_vocab, save_vocab

DEFAULT_LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}


def prepare(
    input_csv: str = "data/Tweets.csv",
    out_dir: str = "data/splits",
    text_col: str = "text",
    label_col: str = "airline_sentiment",
    test_size: float = 0.15,
    val_size_of_train: float = 0.1765,  # ~0.1765 of remaining → overall 15%
    seed: int = 42,
    vocab_min_freq: int = 2,
    lstm_artifacts_dir: str = "experiments/runs/lstm",
):
    """
    - Reads Tweets.csv (US Airline Sentiment format from Kaggle)
    - Keeps only [text_col, label_col]
    - Drops NAs, normalizes labels, filters to {negative, neutral, positive}
    - Stratified split: train / val / test ≈ 70 / 15 / 15
    - Builds vocab from *train* (cleaned) and saves to experiments/runs/lstm/vocab.json
    - Saves splits to data/splits/{train,val,test}.csv
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(lstm_artifacts_dir, exist_ok=True)

    # 1) Load
    df = pd.read_csv(input_csv)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"Expected columns '{text_col}' and '{label_col}' in {input_csv}. "
            f"Found: {list(df.columns)}"
        )

    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    df = df.dropna(subset=["text", "label"]).copy()
    df["label"] = df["label"].astype(str).str.lower().str.strip()

    # 2) Filter to desired classes and map
    allowed = set(DEFAULT_LABEL_MAP.keys())
    before = len(df)
    df = df[df["label"].isin(allowed)].copy()
    after = len(df)
    if after == 0:
        raise ValueError("No rows left after filtering to negative/neutral/positive labels.")
    if after < before:
        print(f"Filtered {before - after} rows with labels outside {allowed}.")

    # 3) Stratified split (test first, then val from remaining)
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["label"]
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_of_train,
        random_state=seed,
        stratify=train_val_df["label"],
    )

    # 4) Save splits
    train_path = os.path.join(out_dir, "airline_train.csv")
    val_path = os.path.join(out_dir, "airline_val.csv")
    test_path = os.path.join(out_dir, "airline_test.csv")
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    # 5) Build vocab from *train* (cleaned text)
    cleaned_train_texts = [clean_tweet(t) for t in train_df["text"].tolist()]
    vocab = build_vocab(cleaned_train_texts, min_freq=vocab_min_freq)
    vocab_path = os.path.join(lstm_artifacts_dir, "vocab.json")
    save_vocab(vocab, vocab_path)

    # 6) Save label map & stats for convenience
    label_map_path = os.path.join(lstm_artifacts_dir, "label_map.json")
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_LABEL_MAP, f, indent=2)

    stats = {
        "seed": seed,
        "sizes": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "class_counts": {
            "train": train_df["label"].value_counts().to_dict(),
            "val": val_df["label"].value_counts().to_dict(),
            "test": test_df["label"].value_counts().to_dict(),
        },
        "vocab_size": len(vocab),
        "paths": {
            "train_csv": train_path,
            "val_csv": val_path,
            "test_csv": test_path,
            "vocab_json": vocab_path,
            "label_map_json": label_map_path,
        },
    }
    with open(os.path.join(out_dir, "prepare_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("✅ Preparation done.")
    print(json.dumps(stats, indent=2))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", type=str, default="data/Tweets.csv")
    ap.add_argument("--out_dir", type=str, default="data/splits")
    ap.add_argument("--text_col", type=str, default="text")
    ap.add_argument("--label_col", type=str, default="airline_sentiment")
    ap.add_argument("--test_size", type=float, default=0.15)
    ap.add_argument("--val_size_of_train", type=float, default=0.1765)  # ~0.1765*0.85 ≈ 0.15
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vocab_min_freq", type=int, default=2)
    ap.add_argument("--lstm_artifacts_dir", type=str, default="experiments/runs/lstm")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare(
        input_csv=args.input_csv,
        out_dir=args.out_dir,
        text_col=args.text_col,
        label_col=args.label_col,
        test_size=args.test_size,
        val_size_of_train=args.val_size_of_train,
        seed=args.seed,
        vocab_min_freq=args.vocab_min_freq,
        lstm_artifacts_dir=args.lstm_artifacts_dir,
    )
