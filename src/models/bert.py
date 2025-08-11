# src/models/bert.py
import torch
import torch.nn as nn
from transformers import AutoModel


class BertClassifier(nn.Module):
    """
    Thin wrapper over a Hugging Face encoder (default: bert-base-uncased)
    for 3-way sentiment classification.

    Forward expects:
        input_ids: LongTensor [B, T]
        attention_mask: LongTensor [B, T]
    Returns:
        logits: FloatTensor [B, 3]
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_classes: int = 3,
        dropout: float = 0.2,
        freeze_encoder: bool = False,
        use_pooler_output: bool = False,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        self.use_pooler_output = use_pooler_output

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        if self.use_pooler_output and hasattr(out, "pooler_output") and out.pooler_output is not None:
            h = out.pooler_output  # [B, H]
        else:
            # CLS token
            h = out.last_hidden_state[:, 0, :]  # [B, H]
        logits = self.classifier(self.dropout(h))  # [B, 3]
        return logits
