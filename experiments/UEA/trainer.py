#!/usr/bin/env python3
"""Trainer for UEA SLinOSS experiments."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm


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
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.logger = logger
        self.grad_clip = grad_clip

    def train_epoch(self, loader: torch.utils.data.DataLoader, epoch: int) -> Tuple[float, float]:
        self.model.train()
        total_loss, total_correct, total = 0.0, 0, 0
        
        pbar = tqdm(loader, desc=f"Train Epoch {epoch}", leave=False)
        for x, lengths, y in pbar:
            x, lengths, y = x.to(self.device), lengths.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(x, lengths)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            
            bsz = y.size(0)
            total_loss += loss.item() * bsz
            total_correct += (logits.argmax(dim=-1) == y).sum().item()
            total += bsz
            pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{total_correct/total:.4f}")
            
        return total_loss / total, total_correct / total

    @torch.no_grad()
    def evaluate(self, loader: torch.utils.data.DataLoader, desc: str = "Eval") -> Tuple[float, float]:
        self.model.eval()
        total_loss, total_correct, total = 0.0, 0, 0
        
        for x, lengths, y in tqdm(loader, desc=desc, leave=False):
            x, lengths, y = x.to(self.device), lengths.to(self.device), y.to(self.device)
            logits = self.model(x, lengths)
            loss = F.cross_entropy(logits, y)
            
            bsz = y.size(0)
            total_loss += loss.item() * bsz
            total_correct += (logits.argmax(dim=-1) == y).sum().item()
            total += bsz
            
        return total_loss / total, total_correct / total
