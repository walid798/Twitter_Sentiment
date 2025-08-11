import pandas as pd
import os
# hanle to go back to the previous foler an acces the file 
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.preprocess import clean_tweet, build_vocab, save_vocab

# Paths
csv_path = "data/Tweets.csv"  # update if needed
out_dir = "experiments/runs/lstm"
os.makedirs(out_dir, exist_ok=True)

# Load data
df = pd.read_csv(csv_path)

# Clean and collect texts
texts = [clean_tweet(t) for t in df["text"].tolist()]

# Build vocab
vocab = build_vocab(texts, min_freq=2)

# Save vocab
save_vocab(vocab, os.path.join(out_dir, "vocab.json"))

print(f"✅ Vocab built with {len(vocab)} tokens and saved to {out_dir}/vocab.json")
