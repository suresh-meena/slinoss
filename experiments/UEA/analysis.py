#!/usr/bin/env python3
"""Analysis script for UEA SLinOSS experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_metrics(history: list, out_dir: Path):
    """Generate beautiful plots for training and validation metrics."""
    df = pd.DataFrame(history)
    sns.set_theme(style="whitegrid", context="talk", palette="muted")
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Loss plot
    sns.lineplot(data=df, x="epoch", y="train_loss", ax=axes[0], label="Train", linewidth=2.5)
    sns.lineplot(data=df, x="epoch", y="val_loss", ax=axes[0], label="Val", linewidth=2.5)
    axes[0].set_title("Loss over Epochs", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross Entropy Loss")
    axes[0].legend()
    
    # Accuracy plot
    sns.lineplot(data=df, x="epoch", y="train_acc", ax=axes[1], label="Train", linewidth=2.5)
    sns.lineplot(data=df, x="epoch", y="val_acc", ax=axes[1], label="Val", linewidth=2.5)
    axes[1].set_title("Accuracy over Epochs", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(out_dir / "metrics_plot.png", dpi=300)
    plt.close()
    print(f"Plot saved to {out_dir / 'metrics_plot.png'}")


def generate_report(summary: dict, out_dir: Path):
    """Generate a detailed text report of the experiment."""
    config = summary["config"]
    history = summary["history"]
    
    report = []
    report.append("=" * 60)
    report.append("UEA SLinOSS EXPERIMENT REPORT")
    report.append("=" * 60)
    report.append(f"Dataset:      {config['dataset']}")
    report.append(f"Run Name:     {config['run_name']}")
    report.append(f"Best Val Acc: {summary['best_val_acc']:.4f}")
    report.append(f"Test Acc:     {summary['test_acc']:.4f}")
    report.append(f"Test Loss:    {summary['test_loss']:.4f}")
    report.append("-" * 60)
    report.append("Hyperparameters:")
    for k, v in config.items():
        if k not in ["dataset", "run_name"]:
            report.append(f"  {k:15}: {v}")
    report.append("-" * 60)
    report.append("Final Epochs (last 5):")
    for h in history[-5:]:
        report.append(
            f"  Epoch {h['epoch']:03d}: train_loss={h['train_loss']:.4f} train_acc={h['train_acc']:.4f} "
            f"val_loss={h['val_loss']:.4f} val_acc={h['val_acc']:.4f}"
        )
    report.append("=" * 60)
    
    report_text = "\n".join(report)
    (out_dir / "report.txt").write_text(report_text)
    print(f"Report saved to {out_dir / 'report.txt'}")
    print(report_text)


def main():
    parser = argparse.ArgumentParser(description="Analyze a UEA experiment run.")
    parser.add_argument("run_dir", type=Path, help="Directory of the run to analyze")
    args = parser.parse_args()

    summary_path = args.run_dir / "summary.json"
    if not summary_path.exists():
        print(f"Error: summary.json not found in {args.run_dir}")
        return

    with summary_path.open("r") as f:
        summary = json.load(f)

    plot_metrics(summary["history"], args.run_dir)
    generate_report(summary, args.run_dir)


if __name__ == "__main__":
    main()
