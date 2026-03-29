"""PyTorch model wrappers for nextchar-style throughput comparison."""

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


class FeedForward(nn.Module):
    def __init__(self, d_model: int, *, mult: int = 4) -> None:
        super().__init__()
        hidden = mult * d_model
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class SLinOSSResidualBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        d_state: int,
        expand: int,
        d_head: int,
        d_conv: int,
        chunk_size: int,
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
        self.ff = FeedForward(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class ContinuousSLinOSSModel(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        block_size: int,
        layers: int,
        d_state: int,
        expand: int,
        d_head: int,
        d_conv: int,
        chunk_size: int,
        normalize_bc: bool,
        backend: str,
    ) -> None:
        super().__init__()
        if int(input_dim) != int(output_dim):
            raise ValueError(
                f"nextchar expects input_dim == output_dim (vocab), got {input_dim} and {output_dim}."
            )
        self.block_size = int(block_size)
        self.token_embed = nn.Embedding(input_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.empty(1, self.block_size, hidden_dim))
        self.blocks = nn.ModuleList(
            [
                SLinOSSResidualBlock(
                    hidden_dim,
                    d_state=d_state,
                    expand=expand,
                    d_head=d_head,
                    d_conv=d_conv,
                    chunk_size=chunk_size,
                    normalize_bc=normalize_bc,
                )
                for _ in range(layers)
            ]
        )
        self.norm_f = nn.RMSNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim, bias=False)
        self.output_proj.weight = self.token_embed.weight
        self.reset_parameters()
        configure_slinoss_backends(self, backend=backend)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected token ids shape (batch, T), got {tuple(x.shape)}")
        if x.shape[1] > self.block_size:
            raise ValueError(
                f"Sequence length {x.shape[1]} exceeds block_size {self.block_size}."
            )
        x = self.token_embed(x)
        x = x + self.pos_embed[:, : x.shape[1], :]
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        return self.output_proj(x)


class Mamba2ResidualBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        d_state: int,
        d_conv: int,
        expand: int,
    ) -> None:
        super().__init__()
        try:
            from mamba_ssm import Mamba2
        except ImportError as exc:
            raise RuntimeError(
                "Mamba2 is unavailable. Install the experiment requirements first."
            ) from exc

        self.norm = nn.RMSNorm(d_model)
        self.mixer = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))


class ContinuousMamba2Model(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        block_size: int,
        layers: int,
        d_state: int,
        d_conv: int,
        expand: int,
    ) -> None:
        super().__init__()
        if int(input_dim) != int(output_dim):
            raise ValueError(
                f"nextchar expects input_dim == output_dim (vocab), got {input_dim} and {output_dim}."
            )
        self.block_size = int(block_size)
        self.token_embed = nn.Embedding(input_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.empty(1, self.block_size, hidden_dim))
        self.blocks = nn.ModuleList(
            [
                Mamba2ResidualBlock(
                    hidden_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(layers)
            ]
        )
        self.norm_f = nn.RMSNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim, bias=False)
        self.output_proj.weight = self.token_embed.weight
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected token ids shape (batch, T), got {tuple(x.shape)}")
        if x.shape[1] > self.block_size:
            raise ValueError(
                f"Sequence length {x.shape[1]} exceeds block_size {self.block_size}."
            )
        x = self.token_embed(x)
        x = x + self.pos_embed[:, : x.shape[1], :]
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        return self.output_proj(x)


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


def count_torch_parameters(model: nn.Module) -> tuple[int, int]:
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    count = sum(int(parameter.numel()) for parameter in params)
    num_bytes = sum(
        int(parameter.numel()) * int(parameter.element_size()) for parameter in params
    )
    return count, num_bytes
