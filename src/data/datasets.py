import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, Dict
from transformers import PreTrainedTokenizerBase

from src.data.preprocess import clean_tweet, encode_text


class TweetDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        text_col: str = "text",
        label_col: str = "label",
        vocab: Optional[Dict[str, int]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        max_len: int = 50,
        label_map: Optional[Dict[str, int]] = None,
        transform: Optional[Callable] = None,
        clean: bool = True,
    ):
        """
        Args:
            csv_path: Path to dataset CSV
            text_col: Column with tweet text
            label_col: Column with sentiment labels
            vocab: For LSTM (word-to-index mapping)
            tokenizer: For BERT
            max_len: Max sequence length
            label_map: Dict mapping label strings to ints
            transform: Optional extra preprocessing function
            clean: Whether to apply clean_tweet()
        """
        self.df = pd.read_csv(csv_path)
        self.text_col = text_col
        self.label_col = label_col
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transform
        self.clean = clean

        # Default label map for US Airline Sentiment
        if label_map is None:
            label_map = {"negative": 0, "neutral": 1, "positive": 2}
        self.label_map = label_map

        # Store as list for speed
        self.texts = self.df[self.text_col].tolist()
        self.labels = [self.label_map[str(l).lower()] for l in self.df[self.label_col]]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.clean:
            text = clean_tweet(text)
        if self.transform:
            text = self.transform(text)

        label = self.labels[idx]

        if self.vocab:
            # LSTM path
            input_ids, attn_mask = encode_text(text, self.vocab, max_len=self.max_len)
            return {
                "input_ids": input_ids.squeeze(0),  # [T]
                "attention_mask": attn_mask.squeeze(0),  # [T]
                "label": torch.tensor(label, dtype=torch.long),
            }
        elif self.tokenizer:
            # BERT path
            enc = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label": torch.tensor(label, dtype=torch.long),
            }
        else:
            raise ValueError("Either vocab or tokenizer must be provided.")
