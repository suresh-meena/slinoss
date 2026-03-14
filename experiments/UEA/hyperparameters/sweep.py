#!/usr/bin/env python3
"""Hyperparameter sweep script for UEA SLinOSS experiments."""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
import time
from pathlib import Path

import torch
import yaml

# Add parent directory to sys.path to import modules from UEA folder
sys.path.append(str(Path(__file__).resolve().parent.parent))

from dataloader import create_dataloaders
from model import UEAClassifier
from trainer import Trainer, EpochMetrics
from utils import set_seed, load_config, get_run_dir, count_parameters


def setup_sweep_logging(sweep_dir: Path) -> logging.Logger:
    sweep_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("UEA_SWEEP")
    logger.setLevel(logging.INFO)
    
    fh = logging.FileHandler(sweep_dir / "sweep.log")
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
    logger.addHandler(fh)
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
    logger.addHandler(ch)
    
    return logger


def main():
    parser = argparse.ArgumentParser(description="Run UEA SLinOSS hyperparameter sweep.")
    parser.add_argument("--base-config", type=Path, default=Path(__file__).resolve().parent.parent / "config.yaml")
    parser.add_argument("--sweep-name", type=str, default=f"sweep_{time.strftime('%Y%m%d-%H%M%S')}")
    parser.add_argument("--datasets", type=str, nargs="+", default=["ArticularyWordRecognition"])
    args = parser.parse_args()

    base_config = load_config(args.base_config)
    sweep_dir = Path("experiments/UEA/runs") / "_sweeps" / args.sweep_name
    logger = setup_sweep_logging(sweep_dir)
    
    # Define sweep grid
    grid = {
        "lr": [1e-4, 3e-4, 1e-3],
        "d_model": [64, 128],
        "n_layers": [2, 3],
        "dropout": [0.1, 0.2]
    }
    
    keys, values = zip(*grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    logger.info(f"Starting sweep '{args.sweep_name}' with {len(combinations)} combinations per dataset.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    results = []
    
    for dataset in args.datasets:
        logger.info(f"=== Dataset: {dataset} ===")
        for i, combo in enumerate(combinations):
            config = base_config.copy()
            config.update(combo)
            config["dataset"] = dataset
            config["run_name"] = f"c{i:03d}_{dataset}"
            
            run_dir = get_run_dir(sweep_dir, dataset, config["run_name"])
            logger.info(f"Run {i+1}/{len(combinations)}: {combo}")
            
            set_seed(config["seed"])
            try:
                loaders, splits = create_dataloaders(config)
                model = UEAClassifier(
                    input_dim=splits.num_features,
                    num_classes=splits.num_classes,
                    **{k: config[k] for k in ["d_model", "n_layers", "d_state", "expand", "d_head", "d_conv", "chunk_size", "dropout", "scan_backend"]}
                ).to(device)
                
                optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
                trainer = Trainer(model, optimizer, device, logger, grad_clip=config["grad_clip"])
                
                best_val_acc = -1.0
                for epoch in range(1, config["epochs"] + 1):
                    trainer.train_epoch(loaders["train"], epoch)
                    _, val_acc = trainer.evaluate(loaders["val"])
                    best_val_acc = max(best_val_acc, val_acc)
                
                logger.info(f"Done. Best Val Acc: {best_val_acc:.4f}")
                results.append({**combo, "dataset": dataset, "best_val_acc": best_val_acc})
                
            except Exception as e:
                logger.error(f"Error in run {i}: {e}")
                continue

    # Save all results
    with (sweep_dir / "results.json").open("w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Sweep complete. Results saved to {sweep_dir / 'results.json'}")


if __name__ == "__main__":
    main()
