#!/usr/bin/env python3
"""Trainer for PPG-DaLiA heart-rate regression."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    train_mae: float
    train_rmse: float
    val_loss: float
    val_mae: float
    val_rmse: float
    elapsed_s: float


class Trainer:
    """Handles training and evaluation loop for PPG tasks (regression)."""
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

    def train_epoch(self, loader: torch.utils.data.DataLoader, epoch: int) -> Tuple[float, float, float]:
        self.model.train()
        total_mse, total_mae, n = 0.0, 0.0, 0
        
        pbar = tqdm(loader, desc=f"Train Epoch {epoch}", leave=False)
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            pred = self.model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            
            bsz = y.size(0)
            total_mse += loss.item() * bsz
            total_mae += F.l1_loss(pred, y).item() * bsz
            n += bsz
            pbar.set_postfix(mse=f"{total_mse/n:.4f}", mae=f"{total_mae/n:.4f}")
            
        return total_mse / n, total_mae / n, math.sqrt(total_mse / n)

    @torch.no_grad()
    def evaluate(self, loader: torch.utils.data.DataLoader, desc: str = "Eval") -> Tuple[float, float, float]:
        self.model.eval()
        total_mse, total_mae, n = 0.0, 0.0, 0
        for x, y in tqdm(loader, desc=desc, leave=False):
            x, y = x.to(self.device), y.to(self.device)
            pred = self.model(x)
            total_mse += F.mse_loss(pred, y).item() * y.size(0)
            total_mae += F.l1_loss(pred, y).item() * y.size(0)
            n += y.size(0)
        return total_mse / n, total_mae / n, math.sqrt(total_mse / n)
