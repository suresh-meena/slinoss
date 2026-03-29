#!/usr/bin/env python3
"""Modular UEA classifier with configurable SLinOSS mixer blocks."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from slinoss.layers import SLinOSSMixer, CuteScanPrepBackend, CuteScanBackend, ReferenceScanPrepBackend, ReferenceScanBackend


class TransposedBatchNorm(nn.Module):
    """Apply BatchNorm1d to (B, L, C) tensors by transposing to (B, C, L)."""

    def __init__(self, d_model: int, track_running_stats: bool = True) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(d_model, affine=True, track_running_stats=track_running_stats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x.transpose(1, 2)).transpose(1, 2)


def get_norm_layer(norm_type: str, d_model: int) -> nn.Module:
    if norm_type == "LayerNorm":
        return nn.LayerNorm(d_model)
    if norm_type == "RMSNorm":
        return nn.RMSNorm(d_model)
    if norm_type == "BatchNorm":
        return TransposedBatchNorm(d_model, track_running_stats=False)
    if norm_type == "BatchNorm_EMA":
        return TransposedBatchNorm(d_model, track_running_stats=True)
    raise ValueError(f"Unknown norm_type: {norm_type}")


class FeedForward(nn.Module):
    """Two-layer feed-forward network with configurable activation."""

    def __init__(
        self,
        d_model: int,
        mult: int = 2,
        dropout: float = 0.0,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        hidden = mult * d_model
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)
        act = activation.lower()
        if act == "gelu":
            self._act = lambda t: F.gelu(t, approximate="tanh")
        elif act == "silu":
            self._act = F.silu
        else:
            raise ValueError(f"Unknown FFN activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._act(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


class MixerBlock(nn.Module):
    """Pre-norm residual SLinOSS block with post-mixer FFN."""

    def __init__(
        self,
        d_model: int,
        d_state: int,
        expand: int,
        d_head: int,
        d_conv: int,
        chunk_size: int,
        dropout: float,
        norm_type: str = "BatchNorm_EMA",
        ffn_activation: str = "silu",
        ffn_mult: int = 2,
        scan_backend: str = "auto",
        **kwargs,
    ) -> None:
        super().__init__()
        mixer_kwargs = {}
        if scan_backend == "cute":
            mixer_kwargs["scanprep_backend"] = CuteScanPrepBackend()
            mixer_kwargs["backend"] = CuteScanBackend()
        elif scan_backend == "reference":
            mixer_kwargs["scanprep_backend"] = ReferenceScanPrepBackend()
            mixer_kwargs["backend"] = ReferenceScanBackend()

        slinoss_keys = {
            "dt_min", "dt_max", "dt_init_floor", "r_min", "r_max",
            "theta_bound", "k_max", "eps", "normalize_bc"
        }
        passed_mixer_kwargs = {k: v for k, v in kwargs.items() if k in slinoss_keys}
        self.norm = get_norm_layer(norm_type, d_model)
        self.mixer = SLinOSSMixer(
            d_model,
            d_state=d_state,
            expand=expand,
            d_head=d_head,
            d_conv=d_conv,
            chunk_size=chunk_size,
            **mixer_kwargs,
            **passed_mixer_kwargs,
        )
        self.drop1 = nn.Dropout(dropout)
        self.ff = FeedForward(d_model, mult=ffn_mult, dropout=dropout, activation=ffn_activation)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        y = self.norm(x)
        y = self.mixer(y)
        y = F.gelu(y, approximate="tanh")
        y = self.drop1(y)
        y = self.ff(y)
        y = self.drop2(y)
        return skip + y


class UEAClassifier(nn.Module):
    """SLinOSS-based classifier for UEA multivariate time series."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int,
        n_layers: int,
        norm_type: str = "BatchNorm_EMA",
        ffn_activation: str = "silu",
        ffn_mult: int = 2,
        **kwargs,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([
            MixerBlock(
                d_model=d_model,
                norm_type=norm_type,
                ffn_activation=ffn_activation,
                ffn_mult=ffn_mult,
                **kwargs,
            )
            for _ in range(n_layers)
        ])
        self.norm = get_norm_layer(norm_type, d_model)
        self.head = nn.Linear(d_model, num_classes)

    def _masked_mean(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        timesteps = x.shape[1]
        idx = torch.arange(timesteps, device=x.device).unsqueeze(0)
        mask = idx < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).to(x.dtype)
        return (x * mask).sum(dim=1) / lengths.clamp_min(1).to(x.dtype).unsqueeze(1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        pooled = self._masked_mean(x, lengths)
        return self.head(pooled)
