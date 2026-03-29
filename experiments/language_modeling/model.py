#!/usr/bin/env python3
"""Model definitions for the FineWeb-Edu language-modeling experiment."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from slinoss.layers import (  # noqa: E402
    AutoCConv1dBackend,
    AutoScanBackend,
    AutoScanPrepBackend,
    CuteScanBackend,
    ReferenceCConv1dBackend,
    ReferenceScanBackend,
    ReferenceScanPrepBackend,
    SLinOSSMixer,
)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SLinOSSBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        d_state: int,
        expand: int,
        d_head: int,
        d_conv: int,
        chunk_size: int,
        ffn_hidden_dim: int,
        normalize_bc: bool,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.mixer = SLinOSSMixer(
            d_model,
            d_state=d_state,
            expand=expand,
            d_head=d_head,
            d_conv=d_conv,
            chunk_size=chunk_size,
            normalize_bc=normalize_bc,
        )
        self.norm2 = nn.RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Mamba2Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        d_state: int,
        d_conv: int,
        expand: int,
        ffn_hidden_dim: int,
    ) -> None:
        super().__init__()
        try:
            from mamba_ssm import Mamba2
        except ImportError as exc:  # pragma: no cover - optional runtime dependency.
            raise RuntimeError(
                "Mamba2 is unavailable. Install experiments/language_modeling/requirements.txt "
                "and ensure mamba-ssm is available."
            ) from exc

        self.norm1 = nn.RMSNorm(d_model)
        self.mixer = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.norm2 = nn.RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class LanguageModel(nn.Module):
    def __init__(self, *, vocab_size: int, config: dict[str, int | bool | str]) -> None:
        super().__init__()
        d_model = int(config["d_model"])
        n_layers = int(config["n_layers"])
        d_state = int(config["d_state"])
        expand = int(config["expand"])
        d_head = int(config["d_head"])
        d_conv = int(config["d_conv"])
        chunk_size = int(config["chunk_size"])
        ffn_hidden_dim = int(config["ffn_hidden_dim"])
        normalize_bc = bool(config.get("normalize_bc", True))
        model_type = str(config["type"])

        self.model_type = model_type
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [
                (
                    SLinOSSBlock(
                        d_model,
                        d_state=d_state,
                        expand=expand,
                        d_head=d_head,
                        d_conv=d_conv,
                        chunk_size=chunk_size,
                        ffn_hidden_dim=ffn_hidden_dim,
                        normalize_bc=normalize_bc,
                    )
                    if model_type == "slinoss"
                    else Mamba2Block(
                        d_model,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                        ffn_hidden_dim=ffn_hidden_dim,
                    )
                )
                for _ in range(n_layers)
            ]
        )
        self.norm_f = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if bool(config.get("tie_embeddings", True)):
            self.lm_head.weight = self.token_embed.weight
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.lm_head:
                nn.init.xavier_uniform_(module.weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(
                f"Expected input_ids shape (batch, seq_len), got {tuple(input_ids.shape)}."
            )
        x = self.token_embed(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        return self.lm_head(x)


def configure_slinoss_backends(model: nn.Module, *, backend: str) -> None:
    if backend not in {"auto", "reference", "cute"}:
        raise ValueError(f"Unsupported SLinOSS backend: {backend}")

    if backend == "auto":
        scan_backend = AutoScanBackend()
        scanprep_backend = AutoScanPrepBackend()
        cconv_backend = AutoCConv1dBackend()
    elif backend == "reference":
        scan_backend = ReferenceScanBackend(compute_dtype=torch.float32)
        scanprep_backend = ReferenceScanPrepBackend()
        cconv_backend = ReferenceCConv1dBackend()
    else:
        scan_backend = CuteScanBackend(compute_dtype=torch.float32)
        scanprep_backend = AutoScanPrepBackend()
        cconv_backend = AutoCConv1dBackend()

    for module in model.modules():
        if isinstance(module, SLinOSSMixer):
            module.backend = scan_backend
            module.scanprep.backend = scanprep_backend
            module.cconv_backend = cconv_backend


def build_model(
    config: dict[str, int | bool | str],
    *,
    vocab_size: int,
    scan_backend: str,
) -> LanguageModel:
    model = LanguageModel(vocab_size=vocab_size, config=config)
    if str(config["type"]) == "slinoss":
        configure_slinoss_backends(model, backend=scan_backend)
    return model
