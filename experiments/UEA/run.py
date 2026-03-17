#!/usr/bin/env python3
"""Main run script for UEA SLinOSS experiments."""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator

from dataloader import create_dataloaders
from model import UEAClassifier
from trainer import EpochMetrics, Trainer
from utils import (
    configure_optimizer,
    count_parameters,
    get_run_dir,
    load_config,
    resolve_mixed_precision,
    set_seed,
    setup_logging,
    specialize_config,
)


def _build_model(config: dict[str, Any], splits, device: torch.device) -> UEAClassifier:
    return UEAClassifier(
        input_dim=splits.num_features,
        num_classes=splits.num_classes,
        **config,
    ).to(device)


def _backend_warmup_or_fallback(
    *,
    model: UEAClassifier,
    config: dict[str, Any],
    loaders,
    splits,
    device: torch.device,
    logger,
) -> UEAClassifier:
    """Try a single no-grad forward and fall back to reference backend if CuTe path fails."""
    scan_backend = str(config.get("scan_backend", "reference"))
    if scan_backend == "reference":
        return model

    try:
        batch = next(iter(loaders["train"]))
        x, lengths, _ = batch
        x = x.to(device)
        lengths = lengths.to(device)
        model.eval()
        with torch.no_grad():
            _ = model(x, lengths)
        model.train()
        return model
    except Exception as exc:
        msg = str(exc)
        fallback_markers = (
            "DSLCudaRuntimeError",
            "cudaErrorInsufficientDriver",
            "error code: 35",
        )
        if scan_backend in {"auto", "cute"} and any(marker in msg for marker in fallback_markers):
            logger.warning(
                "Scan backend '%s' failed warmup with CuTe/CUDA runtime error; "
                "falling back to 'reference'. Error: %s",
                scan_backend,
                msg,
            )
            config["scan_backend"] = "reference"
            return _build_model(config, splits, device)
        raise


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run UEA SLinOSS experiments.")
    p.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--repeat-seeds", action="store_true", help="Run all seeds from config.training.seeds.")
    return p


def _train_step_mode(
    *,
    trainer: Trainer,
    loaders,
    logger,
    run_dir: Path,
    model: UEAClassifier,
    config: dict[str, Any],
    start_time: float,
    accelerator: Accelerator,
) -> tuple[list[EpochMetrics], float]:
    history: list[EpochMetrics] = []
    best_val_acc = -1.0
    no_improvement = 0

    num_steps = int(config.get("num_steps", 0))
    print_steps = int(config.get("print_steps", 1000))
    patience = int(config.get("early_stopping_evals", 10))

    train_iter = iter(loaders["train"])
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for step in range(1, num_steps + 1):
        try:
            x, lengths, y = next(train_iter)
        except StopIteration:
            train_iter = iter(loaders["train"])
            x, lengths, y = next(train_iter)

        train_loss, train_acc, bsz = trainer.train_step(x, lengths, y)
        running_loss += train_loss * bsz
        running_correct += int(train_acc * bsz)
        running_total += bsz

        if step % print_steps != 0:
            continue

        avg_train_loss = running_loss / max(running_total, 1)
        avg_train_acc = running_correct / max(running_total, 1)
        val_loss, val_acc = trainer.evaluate(loaders["val"], desc=f"Val @ step {step}")
        metrics = EpochMetrics(
            epoch=step,
            train_loss=avg_train_loss,
            train_acc=avg_train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            elapsed_s=time.time() - start_time,
        )
        history.append(metrics)

        logger.info(
            "Step %06d: train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f best_val=%.4f",
            step,
            avg_train_loss,
            avg_train_acc,
            val_loss,
            val_acc,
            max(best_val_acc, val_acc),
        )

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            no_improvement = 0
            if accelerator.is_main_process:
                torch.save(
                    {
                        "model_state": accelerator.get_state_dict(model),
                        "config": config,
                        "metrics": [asdict(h) for h in history],
                    },
                    run_dir / "best_model.pt",
                )
        else:
            no_improvement += 1
            if patience > 0 and no_improvement >= patience:
                logger.info("Early stopping after %d evals without improvement.", no_improvement)
                break

    accelerator.wait_for_everyone()
    return history, best_val_acc


def _run_one(config: dict[str, Any], run_name: str, accelerator: Accelerator) -> dict[str, Any]:
    config = dict(config)
    config["run_name"] = run_name
    set_seed(config["seed"])
    device = accelerator.device
    run_dir = Path(config["runs_root"]) / config["dataset"] / run_name
    if accelerator.is_main_process:
        run_dir = get_run_dir(Path(config["runs_root"]), config["dataset"], run_name)
    accelerator.wait_for_everyone()
    logger = setup_logging(run_dir, "train") if accelerator.is_main_process else logging.getLogger("UEA.worker")
    if not accelerator.is_main_process:
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())

    logger.info(f"Starting experiment: {config['dataset']} (Run: {run_name})")
    logger.info(f"Using device: {device}")
    logger.info(f"Config: {json.dumps(config, indent=2)}")

    loaders, splits = create_dataloaders(config)
    logger.info(
        "Dataset splits: Train=%d, Val=%d, Test=%d",
        len(loaders["train"].dataset),
        len(loaders["val"].dataset),
        len(loaders["test"].dataset),
    )

    model = _build_model(config, splits, device)
    if bool(config.get("torch_compile", False)):
        compile_mode = str(config.get("torch_compile_mode", "default"))
        model = torch.compile(model, mode=compile_mode)
    model = _backend_warmup_or_fallback(
        model=model,
        config=config,
        loaders=loaders,
        splits=splits,
        device=device,
        logger=logger,
    )
    logger.info(f"Model parameters: {count_parameters(model)/1e6:.3f}M")

    optimizer = configure_optimizer(
        model,
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
        model,
        optimizer,
        loaders["train"],
        loaders["val"],
        loaders["test"],
    )
    prepared_loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    trainer = Trainer(model, optimizer, device, logger, grad_clip=config["grad_clip"], accelerator=accelerator)

    start_time = time.time()
    if "num_steps" in config:
        history, best_val_acc = _train_step_mode(
            trainer=trainer,
            loaders=prepared_loaders,
            logger=logger,
            run_dir=run_dir,
            model=model,
            config=config,
            start_time=start_time,
            accelerator=accelerator,
        )
    else:
        history = []
        best_val_acc = -1.0
        for epoch in range(1, config["epochs"] + 1):
            train_loss, train_acc = trainer.train_epoch(prepared_loaders["train"], epoch)
            val_loss, val_acc = trainer.evaluate(prepared_loaders["val"], desc=f"Val Epoch {epoch}")

            metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                elapsed_s=time.time() - start_time,
            )
            history.append(metrics)
            logger.info(
                f"Epoch {epoch:03d}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} best_val={max(best_val_acc, val_acc):.4f}"
            )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if accelerator.is_main_process:
                    torch.save(
                        {
                            "model_state": accelerator.get_state_dict(model),
                            "config": config,
                            "metrics": [asdict(h) for h in history],
                        },
                        run_dir / "best_model.pt",
                    )

    logger.info("Evaluating best model on test set...")
    if accelerator.is_main_process and not (run_dir / "best_model.pt").exists():
        torch.save(
            {
                "model_state": accelerator.get_state_dict(model),
                "config": config,
                "metrics": [asdict(h) for h in history],
            },
            run_dir / "best_model.pt",
        )
    accelerator.wait_for_everyone()
    ckpt = torch.load(run_dir / "best_model.pt", weights_only=False, map_location="cpu")
    accelerator.unwrap_model(model).load_state_dict(ckpt["model_state"])
    test_loss, test_acc = trainer.evaluate(prepared_loaders["test"], desc="Test")
    logger.info(f"FINAL TEST ACCURACY: {test_acc:.4f} (Loss: {test_loss:.4f})")

    summary = {
        "config": config,
        "history": [asdict(h) for h in history],
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_loss": test_loss,
    }
    if accelerator.is_main_process:
        with (run_dir / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2)
    accelerator.wait_for_everyone()
    return summary


def main():
    args = build_parser().parse_args()
    base_config = load_config(args.config)
    config = specialize_config(base_config, dataset=args.dataset)
    accelerator = Accelerator(mixed_precision=resolve_mixed_precision(config.get("mixed_precision")))

    run_name = args.run_name if args.run_name else time.strftime("%Y%m%d-%H%M%S")

    if args.repeat_seeds:
        seeds = [int(s) for s in config.get("seeds", [config["seed"]])]
        all_results = []
        for seed in seeds:
            run_cfg = dict(config)
            run_cfg["seed"] = seed
            result = _run_one(run_cfg, f"{run_name}_seed{seed}", accelerator)
            result["seed"] = seed
            all_results.append(result)

        best_vals = [float(r["best_val_acc"]) for r in all_results]
        test_vals = [float(r["test_acc"]) for r in all_results]
        aggregate = {
            "dataset": config["dataset"],
            "run_name": run_name,
            "seeds": seeds,
            "mean_best_val_acc": sum(best_vals) / len(best_vals),
            "mean_test_acc": sum(test_vals) / len(test_vals),
            "runs": all_results,
        }
        summary_path = Path(config["runs_root"]) / config["dataset"] / f"{run_name}_aggregate.json"
        if accelerator.is_main_process:
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with summary_path.open("w") as f:
                json.dump(aggregate, f, indent=2)
    else:
        _run_one(config, run_name, accelerator)


if __name__ == "__main__":
    main()
