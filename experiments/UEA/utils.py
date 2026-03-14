#!/usr/bin/env python3
"""Utilities for UEA experiments."""

from __future__ import annotations

import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def load_config(path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with path.open("r") as f:
        return yaml.safe_load(f)


def setup_logging(log_dir: Path, run_name: str) -> logging.Logger:
    """Configure logging to both console and file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_name}.log"

    logger = logging.getLogger("UEA")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def get_run_dir(base_dir: Path, dataset: str, run_name: Optional[str] = None) -> Path:
    """Generate a unique run directory."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = run_name if run_name else stamp
    run_dir = base_dir / dataset / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
