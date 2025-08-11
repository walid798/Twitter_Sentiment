# src/data/preprocess.py
import re
import json
import emoji
from collections import Counter
from typing import List, Tuple, Dict

import torch


def clean_tweet(text: str) -> str:
    """
    Basic Twitter-aware text cleaning.
    Adjust or extend as needed for your dataset.
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)  # remove urls
    text = re.sub(r"@\w+", " ", text)  # remove mentions
    text = re.sub(r"#", "", text)  # remove hashtag symbol but keep text
    text = emoji.demojize(text)  # convert emojis to text
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces
    return text


def build_vocab(texts: List[str], min_freq: int = 2, specials: List[str] = None) -> Dict[str, int]:
    """
    Build a word-to-index vocab from a list of texts for LSTM.
    """
    if specials is None:
        specials = ["<pad>", "<unk>"]

    counter = Counter()
    for text in texts:
        counter.update(text.split())

    # start vocab with special tokens
    vocab = {tok: idx for idx, tok in enumerate(specials)}

    # add tokens meeting min_freq
    for word, freq in counter.items():
        if freq >= min_freq and word not in vocab:
            vocab[word] = len(vocab)

    return vocab


def save_vocab(vocab: Dict[str, int], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def load_vocab(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def encode_text(text: str, vocab: Dict[str, int], max_len: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode text to tensor of token IDs and attention mask.
    """
    tokens = text.split()
    token_ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]
    token_ids = token_ids[:max_len]
    attn_mask = [1] * len(token_ids)

    # padding
    while len(token_ids) < max_len:
        token_ids.append(vocab["<pad>"])
        attn_mask.append(0)

    return torch.tensor([token_ids]), torch.tensor([attn_mask])
