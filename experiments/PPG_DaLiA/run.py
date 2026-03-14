#!/usr/bin/env python3
"""Main run script for PPG-DaLiA SLinOSS experiments."""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from tqdm.auto import tqdm

from dataloader import create_dataloaders
from model import PPGRegressor
from trainer import Trainer, EpochMetrics
from utils import set_seed, load_config, setup_logging, get_run_dir, count_parameters


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run PPG-DaLiA SLinOSS experiments.")
    p.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    p.add_argument("--run-name", type=str, default=None)
    return p


def main():
    args = build_parser().parse_args()
    config = load_config(args.config)
    
    if args.run_name: config["run_name"] = args.run_name
    else: config["run_name"] = time.strftime("%Y%m%d-%H%M%S")

    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = get_run_dir(Path("experiments/PPG_DaLiA/runs"), config["run_name"])
    logger = setup_logging(run_dir, "train")

    logger.info(f"Starting PPG experiment: {config['run_name']}")
    logger.info(f"Config: {json.dumps(config, indent=2)}")

    loaders, splits = create_dataloaders(config)
    logger.info(f"Splits: Train={len(loaders['train'].dataset)}, Val={len(loaders['val'].dataset)}, Test={len(loaders['test'].dataset)}")

    out_dim = int(splits.train_y.shape[-1])
    model = PPGRegressor(
        input_dim=splits.num_features,
        out_dim=out_dim,
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        d_state=config["d_state"],
        expand=config["expand"],
        d_head=config["d_head"],
        d_conv=config["d_conv"],
        chunk_size=config["chunk_size"],
        dropout=config["dropout"],
        scan_backend=config.get("scan_backend", "auto"),
    ).to(device)
    
    logger.info(f"Model parameters: {count_parameters(model)/1e6:.3f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config["lr"], 
        weight_decay=config["weight_decay"], 
        betas=(0.9, 0.95), 
        fused=device.type == "cuda"
    )
    
    trainer = Trainer(model, optimizer, device, logger, grad_clip=config["grad_clip"])

    history: list[EpochMetrics] = []
    best_val_mae = float("inf")
    start_time = time.time()
    
    for epoch in range(1, config["epochs"] + 1):
        tr_mse, tr_mae, tr_rmse = trainer.train_epoch(loaders["train"], epoch)
        va_mse, va_mae, va_rmse = trainer.evaluate(loaders["val"], desc=f"Val Epoch {epoch}")
        
        metrics = EpochMetrics(
            epoch=epoch, train_loss=tr_mse, train_mae=tr_mae, train_rmse=tr_rmse,
            val_loss=va_mse, val_mae=va_mae, val_rmse=va_rmse,
            elapsed_s=time.time() - start_time
        )
        history.append(metrics)
        
        logger.info(
            f"Epoch {epoch:03d}: tr_mae={tr_mae:.4f} va_mae={va_mae:.4f} best_va_mae={min(best_val_mae, va_mae):.4f}"
        )

        if va_mae < best_val_mae:
            best_val_mae = va_mae
            torch.save({
                "model_state": model.state_dict(),
                "config": config,
                "metrics": [asdict(h) for h in history],
            }, run_dir / "best_model.pt")

    logger.info("Evaluating best model on test set...")
    ckpt = torch.load(run_dir / "best_model.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    te_mse, te_mae, te_rmse = trainer.evaluate(loaders["test"], desc="Test")
    
    logger.info(f"FINAL TEST MAE: {te_mae:.4f} (RMSE: {te_rmse:.4f})")

    summary = {
        "config": config,
        "history": [asdict(h) for h in history],
        "best_val_mae": best_val_mae,
        "test_mae": te_mae,
        "test_rmse": te_rmse,
        "test_mse": te_mse,
    }
    with (run_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
