#!/usr/bin/env python3
"""Trainer for UEA SLinOSS experiments."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm

try:
    from accelerate import Accelerator
except Exception:  # pragma: no cover - optional dependency at import time.
    Accelerator = None  # type: ignore[assignment]


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    elapsed_s: float


class Trainer:
    """Handles training and evaluation loop for UEA tasks."""
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        logger: logging.Logger,
        grad_clip: float = 1.0,
        accelerator: Optional["Accelerator"] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.logger = logger
        self.grad_clip = grad_clip
        self.accelerator = accelerator
        self.amp_enabled = self.accelerator is None and self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16

    def _reduce_stats(self, total_loss: float, total_correct: int, total: int) -> tuple[float, float]:
        stats = torch.tensor(
            [total_loss, float(total_correct), float(total)],
            device=self.device,
            dtype=torch.float64,
        )
        if self.accelerator is not None:
            stats = self.accelerator.reduce(stats, reduction="sum")
        denom = max(float(stats[2].item()), 1.0)
        return float(stats[0].item() / denom), float(stats[1].item() / denom)

    def _forward_loss(self, x: torch.Tensor, lengths: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.accelerator is not None:
            with self.accelerator.autocast():
                logits = self.model(x, lengths)
        else:
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp_enabled,
            ):
                logits = self.model(x, lengths)
        # Keep CE in fp32 for stability when mixer/norm layers run in autocast.
        loss = F.cross_entropy(logits.float(), y)
        return logits, loss

    def train_step(self, x: torch.Tensor, lengths: torch.Tensor, y: torch.Tensor) -> tuple[float, float, int]:
        self.model.train()
        x, lengths, y = x.to(self.device), lengths.to(self.device), y.to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        logits, loss = self._forward_loss(x, lengths, y)
        if self.accelerator is not None:
            self.accelerator.backward(loss)
        else:
            loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        bsz = y.size(0)
        acc = (logits.argmax(dim=-1) == y).sum().item() / bsz
        return float(loss.item()), float(acc), int(bsz)

    def train_epoch(self, loader: torch.utils.data.DataLoader, epoch: int) -> Tuple[float, float]:
        self.model.train()
        total_loss, total_correct, total = 0.0, 0, 0
        
        pbar = tqdm(loader, desc=f"Train Epoch {epoch}", leave=False)
        for x, lengths, y in pbar:
            x, lengths, y = x.to(self.device), lengths.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            logits, loss = self._forward_loss(x, lengths, y)
                
            if self.accelerator is not None:
                self.accelerator.backward(loss)
            else:
                loss.backward()
            
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            
            bsz = y.size(0)
            total_loss += loss.item() * bsz
            total_correct += (logits.argmax(dim=-1) == y).sum().item()
            total += bsz
            pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{total_correct/total:.4f}")
            
        return self._reduce_stats(total_loss, total_correct, total)

    @torch.no_grad()
    def evaluate(self, loader: torch.utils.data.DataLoader, desc: str = "Eval") -> Tuple[float, float]:
        self.model.eval()
        total_loss, total_correct, total = 0.0, 0, 0
        
        for x, lengths, y in tqdm(loader, desc=desc, leave=False):
            x, lengths, y = x.to(self.device), lengths.to(self.device), y.to(self.device)

            logits, loss = self._forward_loss(x, lengths, y)
            
            bsz = y.size(0)
            total_loss += loss.item() * bsz
            total_correct += (logits.argmax(dim=-1) == y).sum().item()
            total += bsz
            
        return self._reduce_stats(total_loss, total_correct, total)
