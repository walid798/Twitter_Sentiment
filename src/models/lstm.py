import torch
import torch.nn as nn
from typing import Optional


class LSTMClassifier(nn.Module):
    """
    Lightweight LSTM sentiment classifier for 3 classes (NEG/NEU/POS).

    Inputs:
        input_ids: LongTensor [B, T]  (indices into vocab)
        attention_mask: LongTensor [B, T] (1 for tokens, 0 for padding)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 200,
        hidden_size: int = 256,
        num_layers: int = 1,
        bidirectional: bool = True,
        num_classes: int = 3,
        dropout: float = 0.3,
        pad_idx: int = 0,
        use_mean_pool: bool = True,
    ):
        super().__init__()
        self.use_mean_pool = use_mean_pool

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=pad_idx
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.0 if num_layers == 1 else dropout,
        )

        lstm_out_dim = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_out_dim, num_classes)

        # Xavier init for linear
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    @staticmethod
    def masked_mean_pool(seq_out: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Mean-pool over time with an attention mask.
        seq_out: [B, T, H]
        mask:    [B, T] with 1 for valid tokens, 0 for pad
        """
        if mask is None:
            return seq_out.mean(dim=1)
        mask = mask.unsqueeze(-1).float()  # [B, T, 1]
        summed = (seq_out * mask).sum(dim=1)      # [B, H]
        counts = mask.sum(dim=1).clamp(min=1.0)   # [B, 1]
        return summed / counts

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Returns:
            logits: [B, num_classes]
        """
        x = self.embedding(input_ids)               # [B, T, E]
        seq_out, _ = self.lstm(x)                   # [B, T, H*D]

        if self.use_mean_pool:
            pooled = self.masked_mean_pool(seq_out, attention_mask)
        else:
            # use last hidden state (concat if bidirectional)
            # seq_out[:, -1] is not strictly equivalent to last hidden
            # when there is padding; mean-pool is safer, but keeping option:
            pooled = seq_out[:, -1, :]

        logits = self.fc(self.dropout(pooled))      # [B, C]
        return logits
