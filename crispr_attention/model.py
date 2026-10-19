"""Bidirectional LSTM with additive attention pooling, for short DNA windows.

Design notes
------------
The vocabulary is six tokens, so the embedding is small. A large
vocabulary would add millions of unused parameters without adding
capacity, because DNA has four letters.

Padding uses index 0 and is excluded from the attention softmax, so
padded positions cannot receive weight. N has its own index, which keeps
real ambiguous bases separate from padding.

Attention weights are returned so that the positions the model draws its
pooled summary from can be inspected. They are not an explanation on
their own. Where the label depends on the identity of one position, the
weights do concentrate there. Where it depends on a comparison between
positions, they need not, and in the deletion notebook they move outside
the informative window from one seed to the next while accuracy holds.
Use ``interpret.saliency`` or ``interpret.integrated_gradients`` to
support a positional claim, and ``interpret.enrichment`` to score it.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .data import VOCAB_SIZE


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal position signal (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].size(1)])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class AttentionBiLSTM(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int = 32, hidden_dim: int = 64,
                 attn_dim: int = 64, dropout: float = 0.3, use_positional: bool = True):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=0)
        self.pos = PositionalEncoding(embed_dim) if use_positional else None
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = nn.Sequential(nn.Linear(hidden_dim * 2, attn_dim), nn.Tanh(),
                                  nn.Linear(attn_dim, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def embed(self, x):
        """Token ids to embeddings. Split out so that attribution methods
        can take gradients with respect to a continuous input."""
        h = self.embedding(x)
        return self.pos(h) if self.pos is not None else h

    def forward_embedded(self, h, mask, return_attention: bool = False):
        h, _ = self.lstm(h)
        scores = self.attn(h).squeeze(-1)                   # [B, L]
        scores = scores.masked_fill(~mask, float("-inf"))   # padding gets no weight
        weights = torch.softmax(scores, dim=1)
        context = self.dropout(torch.sum(weights.unsqueeze(-1) * h, dim=1))
        logits = self.fc(context)
        return (logits, weights) if return_attention else logits

    def forward(self, x, return_attention: bool = False):
        mask = x != 0                                   # True at real bases
        return self.forward_embedded(self.embed(x), mask, return_attention)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
