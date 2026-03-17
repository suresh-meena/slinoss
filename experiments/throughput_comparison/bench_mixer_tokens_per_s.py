#!/usr/bin/env python3
"""Benchmark nextchar train-step throughput by swapping only the mixer."""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
import random
import sys
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from slinoss.layers import SLinOSSMixer  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixer", choices=("slinoss", "mamba2", "both"), default="both")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--d-head", type=int, default=32)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--backend", choices=("reference", "cute"), default="cute")
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    return parser.parse_args()


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sync(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _time_step(fn: Callable[[], None], *, device: torch.device) -> float:
    if device.type == "cuda" and torch.cuda.is_available():
        _sync(device)
        stream = torch.cuda.current_stream(device=device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        fn()
        end.record(stream)
        _sync(device)
        return float(start.elapsed_time(end))

    started = time.perf_counter()
    fn()
    ended = time.perf_counter()
    return (ended - started) * 1000.0


def _configure_slinoss_backend(model: nn.Module, *, backend: str) -> None:
    if backend not in {"reference", "cute"}:
        raise ValueError(f"Unsupported backend: {backend}")
    from slinoss.layers import (  # local import keeps mamba-only runs lightweight
        AutoCConv1dBackend,
        AutoScanPrepBackend,
        CuteScanBackend,
        ReferenceCConv1dBackend,
        ReferenceScanBackend,
        ReferenceScanPrepBackend,
    )

    scan_backend = (
        ReferenceScanBackend(compute_dtype=torch.float32)
        if backend == "reference"
        else CuteScanBackend(compute_dtype=torch.float32)
    )
    scanprep_backend = (
        ReferenceScanPrepBackend() if backend == "reference" else AutoScanPrepBackend()
    )
    cconv_backend = (
        ReferenceCConv1dBackend() if backend == "reference" else AutoCConv1dBackend()
    )
    for module in model.modules():
        if isinstance(module, SLinOSSMixer):
            module.backend = scan_backend
            module.scanprep.backend = scanprep_backend
            module.cconv_backend = cconv_backend


class FeedForward(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        hidden = 4 * d_model
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class SLinOSSMixerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        d_state: int,
        expand: int,
        d_head: int,
        d_conv: int,
        chunk_size: int,
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
            normalize_bc=True,
        )
        self.norm2 = nn.RMSNorm(d_model)
        self.ff = FeedForward(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class Mamba2MixerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        d_state: int,
        expand: int,
        d_conv: int,
    ) -> None:
        super().__init__()
        try:
            from mamba_ssm import Mamba2
        except ImportError as exc:
            raise RuntimeError(
                "mamba_ssm is required for --mixer mamba2/both."
            ) from exc

        self.norm1 = nn.RMSNorm(d_model)
        self.mixer = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.norm2 = nn.RMSNorm(d_model)
        self.ff = FeedForward(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class NextCharMixerLM(nn.Module):
    def __init__(
        self,
        *,
        mixer_name: str,
        vocab_size: int,
        block_size: int,
        d_model: int,
        n_layers: int,
        d_state: int,
        expand: int,
        d_head: int,
        d_conv: int,
        chunk_size: int,
    ) -> None:
        super().__init__()
        self.block_size = int(block_size)
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.empty(1, self.block_size, d_model))
        if mixer_name == "slinoss":
            block_ctor = lambda: SLinOSSMixerBlock(
                d_model,
                d_state=d_state,
                expand=expand,
                d_head=d_head,
                d_conv=d_conv,
                chunk_size=chunk_size,
            )
        elif mixer_name == "mamba2":
            block_ctor = lambda: Mamba2MixerBlock(
                d_model,
                d_state=d_state,
                expand=expand,
                d_conv=d_conv,
            )
        else:
            raise ValueError(f"Unsupported mixer_name: {mixer_name}")

        self.blocks = nn.ModuleList([block_ctor() for _ in range(n_layers)])
        self.norm_f = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.01)
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.lm_head:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.token_embed(idx)
        x = x + self.pos_embed[:, : idx.shape[1], :]
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        return self.lm_head(x)


@dataclass(frozen=True)
class BenchResult:
    mixer: str
    tokens_per_step: int
    warm_tokens_per_s_mean: float
    warm_tokens_per_s_stdev: float
    warm_step_ms_mean: float
    warm_step_ms_stdev: float


def _make_optimizer(model: nn.Module, *, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and "bias" not in name and "norm" not in name.lower():
            decay.append(p)
        else:
            no_decay.append(p)

    use_fused = any(p.is_cuda for p in decay) or any(p.is_cuda for p in no_decay)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
        fused=use_fused,
    )


def _train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    xb: torch.Tensor,
    yb: torch.Tensor,
    *,
    grad_clip: float,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    logits = model(xb)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        tuple(p for p in model.parameters() if p.requires_grad),
        max_norm=grad_clip,
        foreach=xb.device.type == "cuda",
    )
    optimizer.step()


def _build_batches(
    *,
    vocab_size: int,
    batch_size: int,
    block_size: int,
    total_steps: int,
    device: torch.device,
    seed: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(total_steps):
        stream = torch.randint(
            low=0,
            high=vocab_size,
            size=(batch_size, block_size + 1),
            generator=generator,
            dtype=torch.long,
        )
        xb = stream[:, :-1].to(device)
        yb = stream[:, 1:].to(device)
        batches.append((xb, yb))
    return batches


def benchmark_mixer(args: argparse.Namespace, *, mixer_name: str) -> BenchResult:
    device = torch.device(args.device)
    dtype = _dtype_from_name(args.dtype)
    total_steps = int(args.warmup_steps) + int(args.steps)
    batches = _build_batches(
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        block_size=args.block_size,
        total_steps=total_steps,
        device=device,
        seed=args.seed,
    )

    warm_step_repeat_means: list[float] = []
    repeat_tokens_per_s: list[float] = []
    tokens_per_step = int(args.batch_size * args.block_size)

    for rep in range(int(args.repeat)):
        model = NextCharMixerLM(
            mixer_name=mixer_name,
            vocab_size=args.vocab_size,
            block_size=args.block_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            d_state=args.d_state,
            expand=args.expand,
            d_head=args.d_head,
            d_conv=args.d_conv,
            chunk_size=args.chunk_size,
        ).to(device=device, dtype=dtype)

        if mixer_name == "slinoss":
            _configure_slinoss_backend(model, backend=args.backend)

        optimizer = _make_optimizer(
            model,
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )

        for xb, yb in batches[: args.warmup_steps]:
            _train_step(model, optimizer, xb, yb, grad_clip=float(args.grad_clip))

        warm_samples_ms: list[float] = []
        for xb, yb in batches[args.warmup_steps : args.warmup_steps + args.steps]:
            sample_ms = _time_step(
                lambda xb=xb, yb=yb: _train_step(
                    model,
                    optimizer,
                    xb,
                    yb,
                    grad_clip=float(args.grad_clip),
                ),
                device=device,
            )
            warm_samples_ms.append(sample_ms)

        step_mean_ms = float(statistics.fmean(warm_samples_ms))
        warm_step_repeat_means.append(step_mean_ms)
        repeat_tokens_per_s.append((1000.0 * tokens_per_step) / step_mean_ms)

        # de-correlate repeats without changing shape distribution
        _set_seed(args.seed + rep + 1)

    return BenchResult(
        mixer=mixer_name,
        tokens_per_step=tokens_per_step,
        warm_tokens_per_s_mean=float(statistics.fmean(repeat_tokens_per_s)),
        warm_tokens_per_s_stdev=(
            float(statistics.stdev(repeat_tokens_per_s))
            if len(repeat_tokens_per_s) > 1
            else 0.0
        ),
        warm_step_ms_mean=float(statistics.fmean(warm_step_repeat_means)),
        warm_step_ms_stdev=(
            float(statistics.stdev(warm_step_repeat_means))
            if len(warm_step_repeat_means) > 1
            else 0.0
        ),
    )


def main() -> int:
    args = _parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    _set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    mixers = ("slinoss", "mamba2") if args.mixer == "both" else (args.mixer,)
    for mixer_name in mixers:
        result = benchmark_mixer(args, mixer_name=mixer_name)
        print(
            f"mixer={result.mixer} warm_tokens_per_s={result.warm_tokens_per_s_mean:.2f} "
            f"stdev={result.warm_tokens_per_s_stdev:.2f} "
            f"warm_step_ms={result.warm_step_ms_mean:.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
