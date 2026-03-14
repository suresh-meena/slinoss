#!/usr/bin/env python3
"""Main run script for UEA SLinOSS experiments."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from dataloader import create_dataloaders
from model import UEAClassifier
from trainer import EpochMetrics, Trainer
from utils import (
    configure_optimizer,
    count_parameters,
    get_run_dir,
    load_config,
    set_seed,
    setup_logging,
    specialize_config,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run UEA SLinOSS experiments.")
    p.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--dataset", type=str, default=None)
    return p


def main():
    args = build_parser().parse_args()
    base_config = load_config(args.config)
    config = specialize_config(base_config, dataset=args.dataset)

    if args.run_name:
        config["run_name"] = args.run_name
    else:
        config["run_name"] = time.strftime("%Y%m%d-%H%M%S")

    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = get_run_dir(Path(config["runs_root"]), config["dataset"], config["run_name"])
    logger = setup_logging(run_dir, "train")

    logger.info(f"Starting experiment: {config['dataset']} (Run: {config['run_name']})")
    logger.info(f"Using device: {device}")
    logger.info(f"Config: {json.dumps(config, indent=2)}")

    loaders, splits = create_dataloaders(config)
    logger.info(f"Dataset splits: Train={len(loaders['train'].dataset)}, Val={len(loaders['val'].dataset)}, Test={len(loaders['test'].dataset)}")

    model = UEAClassifier(
        input_dim=splits.num_features,
        num_classes=splits.num_classes,
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

    optimizer = configure_optimizer(
        model,
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    trainer = Trainer(model, optimizer, device, logger, grad_clip=config["grad_clip"])

    history: list[EpochMetrics] = []
    best_val_acc = -1.0
    start_time = time.time()
    
    for epoch in range(1, config["epochs"] + 1):
        train_loss, train_acc = trainer.train_epoch(loaders["train"], epoch)
        val_loss, val_acc = trainer.evaluate(loaders["val"], desc=f"Val Epoch {epoch}")
        
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            elapsed_s=time.time() - start_time
        )
        history.append(metrics)
        
        logger.info(
            f"Epoch {epoch:03d}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} best_val={max(best_val_acc, val_acc):.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "config": config,
                "metrics": [asdict(h) for h in history],
            }, run_dir / "best_model.pt")

    # Final test
    logger.info("Evaluating best model on test set...")
    ckpt = torch.load(run_dir / "best_model.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_acc = trainer.evaluate(loaders["test"], desc="Test")
    
    logger.info(f"FINAL TEST ACCURACY: {test_acc:.4f} (Loss: {test_loss:.4f})")

    # Save summary
    summary = {
        "config": config,
        "history": [asdict(h) for h in history],
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_loss": test_loss,
    }
    with (run_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
