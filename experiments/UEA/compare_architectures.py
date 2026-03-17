#!/usr/bin/env python3
"""Activation and normalization comparison script for UEA SLinOSS experiments."""

import time
import logging
import pandas as pd
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from slinoss.layers import SLinOSSMixer, CuteScanPrepBackend, CuteScanBackend
from dataloader import create_dataloaders
from trainer import Trainer
from utils import load_config, specialize_config, configure_optimizer, count_parameters, set_seed


class TransposedBatchNorm(nn.Module):
    """Wrapper to apply BatchNorm1d to (B, L, C) inputs."""
    def __init__(self, d_model, track_running_stats=True):
        super().__init__()
        # affine=False to match the original DLinOSS reference, but we can set to True as well.
        # We will use affine=True here to make it comparable to LayerNorm/RMSNorm which have learnable parameters.
        self.bn = nn.BatchNorm1d(d_model, affine=True, track_running_stats=track_running_stats)

    def forward(self, x):
        return self.bn(x.transpose(1, 2)).transpose(1, 2)


def get_norm_layer(norm_type, d_model):
    if norm_type == "LayerNorm":
        return nn.LayerNorm(d_model)
    elif norm_type == "RMSNorm":
        return nn.RMSNorm(d_model)
    elif norm_type == "BatchNorm":
        return TransposedBatchNorm(d_model, track_running_stats=False)
    elif norm_type == "BatchNorm_EMA":
        return TransposedBatchNorm(d_model, track_running_stats=True)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")


class FFN_Gelu(nn.Module):
    def __init__(self, d_model, mult=2):
        super().__init__()
        hidden = int(d_model * mult)
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))
    
class FFN_Silu(nn.Module):
    def __init__(self, d_model, mult=2):
        super().__init__()
        hidden = int(d_model * mult)
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)

    def forward(self, x):
        return self.fc2(F.silu(self.fc1(x)))

def build_mixer(config):
    return SLinOSSMixer(
        d_model=config["d_model"],
        d_state=config["d_state"],
        expand=config["expand"],
        d_head=config["d_head"],
        d_conv=config["d_conv"],
        chunk_size=config["chunk_size"],
        scanprep_backend=CuteScanPrepBackend(),
        backend=CuteScanBackend(),
        normalize_bc=bool(config.get("normalize_bc", True)),
    )


def build_activation(activation_type, d_model, ffn_mult=2):
    if activation_type == "FFN_Gelu":
        return FFN_Gelu(d_model, mult=ffn_mult)
    if activation_type == "FFN_Silu":
        return FFN_Silu(d_model, mult=ffn_mult)
    raise ValueError(f"Unknown activation_type: {activation_type}")


class SLinOSSMixerActivationBlock(nn.Module):
    """SLinOSSMixer residual block with pluggable post-mixer transform."""

    def __init__(self, config, norm_type, activation_type, ffn_mult=2):
        super().__init__()
        self.norm = get_norm_layer(norm_type, config["d_model"])
        self.mixer = build_mixer(config)
        self.drop1 = nn.Dropout(config["dropout"])
        self.activation = build_activation(
            activation_type,
            config["d_model"],
            ffn_mult=ffn_mult,
        )
        self.drop2 = nn.Dropout(config["dropout"])

    def forward(self, x):
        skip = x
        y = self.norm(x)
        y = self.mixer(y)
        y = F.gelu(y, approximate="tanh")
        y = self.drop1(y)
        y = self.activation(y)
        y = self.drop2(y)
        return skip + y

class VariantClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        config,
        activation_type,
        norm_type,
        ffn_mult=2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, config["d_model"])

        self.blocks = nn.ModuleList([
            SLinOSSMixerActivationBlock(
                config,
                norm_type,
                activation_type,
                ffn_mult=ffn_mult,
            )
            for _ in range(config["n_layers"])
        ])
        self.norm = get_norm_layer(norm_type, config["d_model"])
        self.head = nn.Linear(config["d_model"], num_classes)

    def _masked_mean(self, x, lengths):
        timesteps = x.shape[1]
        idx = torch.arange(timesteps, device=x.device).unsqueeze(0)
        mask = idx < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).to(x.dtype)
        return (x * mask).sum(dim=1) / lengths.clamp_min(1).to(x.dtype).unsqueeze(1)

    def forward(self, x, lengths):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        pooled = self._masked_mean(x, lengths)
        return self.head(pooled)

def main():
    base_config = load_config(Path("experiments/UEA/config.yaml"))
    
    # The 6 active datasets
    datasets = [
        "EigenWorms", "SelfRegulationSCP1", "SelfRegulationSCP2", 
        "EthanolConcentration", "Heartbeat", "MotorImagery"
    ]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Running on device: {device} using model dtype {model_dtype}")

    activations = ["FFN_Gelu", "FFN_Silu"]
    normalizations = ["LayerNorm", "RMSNorm", "BatchNorm", "BatchNorm_EMA"]
    num_runs = 3
    ffn_mult = 2

    print(
        "FFN variants: "
        f"mult={ffn_mult}, variants={activations}"
    )

    logger = logging.getLogger("arch_compare")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    
    results = []
    
    for dataset in datasets:
        print(f"\n==========================================")
        print(f"=== TESTING DATASET: {dataset.upper()} ===")
        print(f"==========================================")
        
        config = specialize_config(base_config, dataset=dataset)
        # 15 epochs for quick ablation
        config["epochs"] = 15
        seeds = [config["seed"] + i for i in range(num_runs)]
        print(f"Using seeds: {seeds}")
        
        for activation in activations:
            for norm in normalizations:
                print(f"\n--- {dataset} | Activation: {activation} | Norm: {norm} ---")
                seed_best_vals = []
                seed_times = []
                successful_seeds = []
                params = None

                for seed in seeds:
                    print(f"Seed {seed}")
                    run_config = dict(config)
                    run_config["seed"] = seed
                    try:
                        set_seed(seed)
                        loaders, splits = create_dataloaders(run_config)

                        model = VariantClassifier(
                            splits.num_features,
                            splits.num_classes,
                            run_config,
                            activation,
                            norm,
                            ffn_mult=ffn_mult,
                        ).to(device=device, dtype=model_dtype)

                        # Warmup CUTE (Forward + Backward to trigger JIT compilation before timing)
                        batch = next(iter(loaders["train"]))
                        x, lengths, y = batch
                        model.train()
                        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                            logits = model(x.to(device), lengths.to(device))
                            loss = F.cross_entropy(logits, y.to(device))
                        loss.backward()
                        model.zero_grad(set_to_none=True)

                        if params is None:
                            params = count_parameters(model)
                            print(f"Parameters: {params:,}")

                        optimizer = configure_optimizer(
                            model,
                            lr=run_config["lr"],
                            weight_decay=run_config["weight_decay"],
                        )

                        trainer = Trainer(model, optimizer, device, logger, grad_clip=run_config["grad_clip"])

                        best_val = 0.0
                        start_time = time.time()
                        for epoch in range(1, run_config["epochs"] + 1):
                            train_loss, train_acc = trainer.train_epoch(loaders["train"], epoch)
                            val_loss, val_acc = trainer.evaluate(loaders["val"], desc="")
                            if val_acc > best_val:
                                best_val = val_acc
                        elapsed = time.time() - start_time

                        print(f"Best Val Acc: {best_val:.4f} | Time: {elapsed:.2f}s")
                        seed_best_vals.append(best_val)
                        seed_times.append(elapsed)
                        successful_seeds.append(seed)
                    except Exception as e:
                        print(
                            f"Run failed for {dataset} | {activation} | {norm} | seed {seed}: {e}"
                        )
                        continue

                if len(seed_best_vals) != num_runs:
                    print(
                        f"Skipping {dataset} | {activation} | {norm}: "
                        f"need {num_runs} successful runs, got {len(seed_best_vals)}"
                    )
                    continue

                avg_best_val = sum(seed_best_vals) / len(seed_best_vals)
                avg_time = sum(seed_times) / len(seed_times)
                print(
                    f"Average over {num_runs} seeds -> "
                    f"Best Val Acc: {avg_best_val:.4f} | Time: {avg_time:.2f}s"
                )
                results.append({
                    "Dataset": dataset,
                    "Architecture": activation,
                    "Normalization": norm,
                    "Parameters": params,
                    "Seeds": ",".join(str(s) for s in successful_seeds),
                    "Seed Count": len(successful_seeds),
                    "Best Val Acc": round(avg_best_val, 4),
                    "Time (s)": round(avg_time, 2)
                })
        
    df = pd.DataFrame(results)
    if df.empty:
        print("\nNo valid results (all cells failed 3-run requirement).")
        return

    print("\n=== FINAL ABLATION RESULTS (LONG FORMAT) ===")
    print(df.to_string(index=False))

    print("\n=== TABLE: ARCHITECTURE VS NORMALIZATION (MEAN BEST VAL ACC) ===")
    overall_acc = (
        df.groupby(["Architecture", "Normalization"], as_index=False)["Best Val Acc"]
        .mean()
        .pivot(index="Architecture", columns="Normalization", values="Best Val Acc")
    )
    print(overall_acc.round(4).to_string())

    print("\n=== TABLE: ARCHITECTURE VS NORMALIZATION (MEAN TIME S) ===")
    overall_time = (
        df.groupby(["Architecture", "Normalization"], as_index=False)["Time (s)"]
        .mean()
        .pivot(index="Architecture", columns="Normalization", values="Time (s)")
    )
    print(overall_time.round(2).to_string())

    for dataset in datasets:
        df_ds = df[df["Dataset"] == dataset]
        if df_ds.empty:
            continue
        print(f"\n=== {dataset}: ARCHITECTURE VS NORMALIZATION (BEST VAL ACC) ===")
        table_ds = df_ds.pivot(index="Architecture", columns="Normalization", values="Best Val Acc")
        print(table_ds.round(4).to_string())

if __name__ == "__main__":
    main()