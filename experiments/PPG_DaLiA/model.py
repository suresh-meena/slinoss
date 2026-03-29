#!/usr/bin/env python3
"""Modular PPG Regressor model using SLinOSS mixer."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from slinoss.layers import SLinOSSMixer, CuteScanPrepBackend, CuteScanBackend, ReferenceScanPrepBackend, ReferenceScanBackend


class FeedForward(nn.Module):
    """GELU feed-forward network."""
    def __init__(self, d_model: int, mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = mult * d_model
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x), approximate="tanh")
        x = self.drop(x)
        return self.fc2(x)


class MixerBlock(nn.Module):
    """Mixer block with SLinOSS and FeedForward."""
    def __init__(
        self,
        d_model: int,
        d_state: int,
        expand: int,
        d_head: int,
        d_conv: int,
        chunk_size: int,
        dropout: float,
        scan_backend: str = "auto",
    ) -> None:
        super().__init__()
        mixer_kwargs = {}
        if scan_backend == "cute":
            mixer_kwargs["scanprep_backend"] = CuteScanPrepBackend()
            mixer_kwargs["backend"] = CuteScanBackend()
        elif scan_backend == "reference":
            mixer_kwargs["scanprep_backend"] = ReferenceScanPrepBackend()
            mixer_kwargs["backend"] = ReferenceScanBackend()

        self.norm1 = nn.RMSNorm(d_model)
        self.mixer = SLinOSSMixer(
            d_model,
            d_state=d_state,
            expand=expand,
            d_head=d_head,
            d_conv=d_conv,
            chunk_size=chunk_size,
            normalize_bc=True,
            **mixer_kwargs,
        )
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.RMSNorm(d_model)
        self.ff = FeedForward(d_model, mult=4, dropout=dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop1(self.mixer(self.norm1(x)))
        x = x + self.drop2(self.ff(self.norm2(x)))
        return x


class PPGRegressor(nn.Module):
    """SLinOSS-based regressor for heart rate estimation from PPG."""
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        d_model: int,
        n_layers: int,
        d_state: int,
        expand: int,
        d_head: int,
        d_conv: int,
        chunk_size: int,
        dropout: float,
        scan_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([
            MixerBlock(d_model, d_state, expand, d_head, d_conv, chunk_size, dropout, scan_backend)
            for _ in range(n_layers)
        ])
        self.norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        # For PPG-DaLiA, we usually pool across time or take the final state
        # The existing script uses mean pooling.
        pooled = x.mean(dim=1)
        return self.head(pooled)
