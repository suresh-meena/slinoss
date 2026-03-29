#!/usr/bin/env python3
"""FineWeb-Edu language-model pretraining with SLinOSS or Mamba2 blocks."""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from torch.nn import functional as F

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from data import create_dataloaders, load_tokenizer
from model import build_model
from utils import (
    TrainState,
    append_jsonl,
    apply_overrides,
    build_cosine_scheduler,
    configure_optimizer,
    count_parameters,
    create_run_name,
    derive_training_steps,
    ensure_dir,
    load_config,
    resolve_config,
    save_json,
    set_seed,
    setup_logging,
    validate_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a SLinOSS language model on FineWeb-Edu with Hugging Face Accelerate."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config.yaml",
    )
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config values with dotted assignments like training.max_steps=100.",
    )
    return parser


def _resolve_mixed_precision(value: object) -> str:
    if value is None:
        return "bf16" if torch.cuda.is_available() else "no"
    if isinstance(value, bool):
        return "bf16" if value else "no"
    resolved = str(value).strip().lower()
    if resolved in {"false", "no", "off", "none"}:
        return "no"
    if resolved in {"true", "on"}:
        return "bf16" if torch.cuda.is_available() else "no"
    if resolved not in {"no", "fp8", "fp16", "bf16"}:
        raise ValueError(f"Unknown mixed_precision mode: {value}")
    return resolved


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device=device, non_blocking=True)
        for key, value in batch.items()
    }


def _compute_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    vocab_size = logits.shape[-1]
    return F.cross_entropy(
        logits.reshape(-1, vocab_size).float(),
        labels.reshape(-1),
    )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    *,
    accelerator: Accelerator,
    max_batches: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()

    total_loss = torch.zeros(2, device=accelerator.device, dtype=torch.float64)
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        batch = _move_batch(batch, accelerator.device)
        logits = model(batch["input_ids"])
        labels = batch["labels"]
        loss_sum = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            labels.reshape(-1),
            reduction="sum",
        )
        stats = torch.tensor(
            [float(loss_sum), float(labels.numel())],
            device=accelerator.device,
            dtype=torch.float64,
        )
        total_loss += accelerator.reduce(stats, reduction="sum")

    total_tokens = max(float(total_loss[1].item()), 1.0)
    mean_nll = float(total_loss[0].item() / total_tokens)
    ppl = math.exp(min(mean_nll, 20.0))

    if was_training:
        model.train()
    return {
        "nll": mean_nll,
        "ppl": ppl,
    }


def _backend_warmup_or_fallback(
    *,
    model: torch.nn.Module,
    config: dict[str, Any],
    vocab_size: int,
    device: torch.device,
    logger: logging.Logger,
) -> torch.nn.Module:
    model_cfg = dict(config["model"])
    if str(model_cfg["type"]) != "slinoss":
        return model

    backend = str(config["backend"].get("scan_backend", "reference"))
    if backend == "reference":
        return model

    try:
        seq_len = min(int(config["training"]["seq_len"]), int(model_cfg["chunk_size"]) * 2)
        dummy = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long, device=device)
        model.eval()
        with torch.no_grad():
            _ = model(dummy)
        model.train()
        return model
    except Exception as exc:
        msg = str(exc)
        fallback_markers = (
            "DSLCudaRuntimeError",
            "cudaErrorInsufficientDriver",
            "error code: 35",
        )
        if backend in {"auto", "cute"} and any(marker in msg for marker in fallback_markers):
            logger.warning(
                "SLinOSS backend '%s' failed warmup; falling back to reference. Error: %s",
                backend,
                msg,
            )
            config["backend"]["scan_backend"] = "reference"
            fallback = build_model(
                config["model"],
                vocab_size=vocab_size,
                scan_backend="reference",
            )
            return fallback.to(device)
        raise


def _save_checkpoint(
    *,
    accelerator: Accelerator,
    run_dir: Path,
    tag: str,
    train_state: TrainState,
    extra_state: dict[str, Any],
) -> Path:
    checkpoint_dir = run_dir / "checkpoints" / tag
    accelerator.wait_for_everyone()
    accelerator.save_state(str(checkpoint_dir))
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        payload = {
            "tag": tag,
            "global_step": int(train_state.global_step),
            "seen_batches": int(train_state.seen_batches),
            "tokens_seen": int(train_state.tokens_seen),
            **extra_state,
        }
        save_json(checkpoint_dir / "meta.json", payload)
    return checkpoint_dir


def _write_summary(
    *,
    run_dir: Path,
    run_name: str,
    config: dict[str, Any],
    parameter_count: tuple[int, float],
    train_state: TrainState,
    derived: dict[str, int],
    best_eval: dict[str, float] | None,
    last_eval: dict[str, float] | None,
) -> None:
    payload = {
        "run_name": run_name,
        "global_step": int(train_state.global_step),
        "seen_batches": int(train_state.seen_batches),
        "tokens_seen": int(train_state.tokens_seen),
        "elapsed_s": float(train_state.elapsed_s),
        "parameter_count": int(parameter_count[0]),
        "parameter_count_m": float(parameter_count[1]),
        "derived": derived,
        "model": dict(config["model"]),
        "backend": dict(config["backend"]),
        "data": {
            key: value
            for key, value in dict(config["data"]).items()
            if key not in {"cache_dir"}
        },
        "training": {
            key: value
            for key, value in dict(config["training"]).items()
            if key not in {"resume_from"}
        },
        "best_eval": best_eval,
        "last_eval": last_eval,
    }
    save_json(run_dir / "summary.json", payload)


def main() -> None:
    args = build_parser().parse_args()
    raw_config = load_config(args.config)
    config = resolve_config(raw_config, preset=args.preset)
    config = apply_overrides(config, args.overrides)
    if args.resume_from is not None:
        config.setdefault("training", {})
        config["training"]["resume_from"] = str(args.resume_from)
    validate_config(config)

    output_root = ensure_dir(Path(config["experiment"]["output_root"]).resolve())
    mixed_precision = _resolve_mixed_precision(config["training"].get("mixed_precision"))
    accelerator = Accelerator(
        gradient_accumulation_steps=int(config["training"]["gradient_accumulation_steps"]),
        mixed_precision=mixed_precision,
        project_config=ProjectConfiguration(
            project_dir=str(output_root),
            logging_dir=str(output_root / "logs"),
            automatic_checkpoint_naming=False,
        ),
        step_scheduler_with_optimizer=False,
    )

    set_seed(int(config["experiment"]["seed"]))
    run_name = create_run_name(config, run_name=args.run_name)
    run_dir = output_root / run_name
    if accelerator.is_main_process:
        ensure_dir(run_dir)
        ensure_dir(run_dir / "checkpoints")
    accelerator.wait_for_everyone()

    logger = setup_logging(run_dir, main_process=accelerator.is_main_process)
    if accelerator.is_main_process:
        save_json(run_dir / "config.resolved.json", config)

    logger.info("Starting run: %s", run_name)
    logger.info("Accelerate processes=%d mixed_precision=%s", accelerator.num_processes, mixed_precision)
    logger.info("Model preset=%s type=%s", config["experiment"].get("preset"), config["model"]["type"])

    with accelerator.main_process_first():
        tokenizer = load_tokenizer(dict(config["data"]))

    train_loader, eval_loader, data_spec = create_dataloaders(
        config,
        tokenizer=tokenizer,
        process_index=accelerator.process_index,
        num_processes=accelerator.num_processes,
        device_type=accelerator.device.type,
    )
    logger.info(
        "Dataset=%s/%s tokenizer=%s vocab=%d",
        config["data"]["dataset_name"],
        config["data"].get("dataset_config_name"),
        data_spec.tokenizer_name,
        data_spec.vocab_size,
    )

    model = build_model(
        config["model"],
        vocab_size=data_spec.vocab_size,
        scan_backend=str(config["backend"]["scan_backend"]),
    ).to(accelerator.device)
    model = _backend_warmup_or_fallback(
        model=model,
        config=config,
        vocab_size=data_spec.vocab_size,
        device=accelerator.device,
        logger=logger,
    )
    if bool(config["training"].get("torch_compile", False)):
        compile_mode = str(config["training"].get("torch_compile_mode", "default"))
        model = torch.compile(model, mode=compile_mode)

    parameter_count = count_parameters(model)
    logger.info(
        "Parameter count: %d trainable (%.3fM), target=%d",
        parameter_count[0],
        parameter_count[1],
        int(config["model"]["target_params"]),
    )

    derived = derive_training_steps(
        config,
        world_size=accelerator.num_processes,
    )
    logger.info(
        "Derived tokens_per_step=%d max_steps=%d",
        derived["tokens_per_step"],
        derived["max_steps"],
    )

    optimizer = configure_optimizer(
        model,
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        betas=(
            float(config["training"]["adam_beta1"]),
            float(config["training"]["adam_beta2"]),
        ),
        eps=float(config["training"]["adam_eps"]),
    )
    scheduler = build_cosine_scheduler(
        optimizer,
        warmup_steps=int(config["training"]["warmup_steps"]),
        total_steps=int(derived["max_steps"]),
        min_lr_ratio=float(config["training"]["min_lr_ratio"]),
    )
    train_state = TrainState()
    accelerator.register_for_checkpointing(scheduler)
    accelerator.register_for_checkpointing(train_state)

    model, optimizer = accelerator.prepare(model, optimizer)

    resume_from = config["training"].get("resume_from")
    if resume_from:
        resume_path = Path(str(resume_from)).resolve()
        logger.info("Resuming from checkpoint: %s", resume_path)
        accelerator.load_state(str(resume_path))
        train_loader = accelerator.skip_first_batches(train_loader, num_batches=train_state.seen_batches)

    if train_state.global_step >= int(derived["max_steps"]):
        logger.info("Checkpoint is already at or beyond max_steps=%d.", int(derived["max_steps"]))

    metrics_path = run_dir / "metrics.jsonl"
    wall_start = time.time()
    running_loss_numer = 0.0
    running_tokens = 0.0
    last_eval: dict[str, float] | None = None
    best_eval: dict[str, float] | None = None

    for batch in train_loader:
        if train_state.global_step >= int(derived["max_steps"]):
            break

        batch = _move_batch(batch, accelerator.device)
        model.train()
        with accelerator.accumulate(model):
            logits = model(batch["input_ids"])
            loss = _compute_loss(logits, batch["labels"])
            accelerator.backward(loss)
            if accelerator.sync_gradients and float(config["training"]["grad_clip"]) > 0.0:
                accelerator.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float(config["training"]["grad_clip"]),
                )
            optimizer.step()
            if accelerator.sync_gradients:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        local_tokens = batch["labels"].numel()
        token_stats = torch.tensor(
            [float(loss.detach()) * float(local_tokens), float(local_tokens)],
            device=accelerator.device,
            dtype=torch.float64,
        )
        token_stats = accelerator.reduce(token_stats, reduction="sum")
        running_loss_numer += float(token_stats[0].item())
        running_tokens += float(token_stats[1].item())
        train_state.tokens_seen += int(token_stats[1].item())
        train_state.seen_batches += 1

        if not accelerator.sync_gradients:
            continue

        train_state.global_step += 1
        train_state.last_train_loss = running_loss_numer / max(running_tokens, 1.0)
        train_state.elapsed_s += time.time() - wall_start
        wall_start = time.time()

        if train_state.global_step % int(config["training"]["log_interval"]) == 0 and accelerator.is_main_process:
            lr = float(scheduler.get_last_lr()[0])
            log_payload = {
                "event": "train",
                "global_step": int(train_state.global_step),
                "seen_batches": int(train_state.seen_batches),
                "tokens_seen": int(train_state.tokens_seen),
                "train_loss": float(train_state.last_train_loss),
                "lr": lr,
                "elapsed_s": float(train_state.elapsed_s),
            }
            logger.info(
                "step=%07d loss=%.4f lr=%.6g tokens=%d",
                train_state.global_step,
                train_state.last_train_loss,
                lr,
                train_state.tokens_seen,
            )
            append_jsonl(metrics_path, log_payload)
            running_loss_numer = 0.0
            running_tokens = 0.0

        should_eval = (
            train_state.global_step % int(config["training"]["eval_interval"]) == 0
            or train_state.global_step == int(derived["max_steps"])
        )
        if should_eval:
            last_eval = evaluate(
                model,
                eval_loader,
                accelerator=accelerator,
                max_batches=int(config["training"]["max_eval_batches"]),
            )
            if last_eval["nll"] <= train_state.best_val_nll:
                train_state.best_val_nll = float(last_eval["nll"])
                train_state.best_val_ppl = float(last_eval["ppl"])
                best_eval = dict(last_eval)
            if accelerator.is_main_process:
                payload = {
                    "event": "eval",
                    "global_step": int(train_state.global_step),
                    "tokens_seen": int(train_state.tokens_seen),
                    "val_nll": float(last_eval["nll"]),
                    "val_ppl": float(last_eval["ppl"]),
                    "best_val_nll": float(train_state.best_val_nll),
                    "best_val_ppl": float(train_state.best_val_ppl),
                    "elapsed_s": float(train_state.elapsed_s),
                }
                logger.info(
                    "eval step=%07d val_nll=%.4f val_ppl=%.4f best_ppl=%.4f",
                    train_state.global_step,
                    last_eval["nll"],
                    last_eval["ppl"],
                    train_state.best_val_ppl,
                )
                append_jsonl(metrics_path, payload)
                _write_summary(
                    run_dir=run_dir,
                    run_name=run_name,
                    config=config,
                    parameter_count=parameter_count,
                    train_state=train_state,
                    derived=derived,
                    best_eval=best_eval,
                    last_eval=last_eval,
                )

        should_save = (
            train_state.global_step % int(config["training"]["save_interval"]) == 0
            or train_state.global_step == int(derived["max_steps"])
        )
        if should_save:
            checkpoint_dir = _save_checkpoint(
                accelerator=accelerator,
                run_dir=run_dir,
                tag=f"step_{train_state.global_step:07d}",
                train_state=train_state,
                extra_state={
                    "val_nll": None if last_eval is None else float(last_eval["nll"]),
                    "val_ppl": None if last_eval is None else float(last_eval["ppl"]),
                },
            )
            if accelerator.is_main_process:
                logger.info("Saved checkpoint: %s", checkpoint_dir)

    accelerator.wait_for_everyone()
    if last_eval is None:
        last_eval = evaluate(
            model,
            eval_loader,
            accelerator=accelerator,
            max_batches=int(config["training"]["max_eval_batches"]),
        )
        if best_eval is None:
            best_eval = dict(last_eval)
            train_state.best_val_nll = float(last_eval["nll"])
            train_state.best_val_ppl = float(last_eval["ppl"])

    if accelerator.is_main_process:
        _write_summary(
            run_dir=run_dir,
            run_name=run_name,
            config=config,
            parameter_count=parameter_count,
            train_state=train_state,
            derived=derived,
            best_eval=best_eval,
            last_eval=last_eval,
        )
        logger.info(
            "Finished run: steps=%d tokens=%d best_val_ppl=%.4f",
            train_state.global_step,
            train_state.tokens_seen,
            train_state.best_val_ppl,
        )


if __name__ == "__main__":
    main()
