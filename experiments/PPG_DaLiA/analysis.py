#!/usr/bin/env python3
"""Analysis script for PPG-DaLiA SLinOSS experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_metrics(history: list, out_dir: Path):
    """Generate plots for PPG regression metrics."""
    df = pd.DataFrame(history)
    sns.set_theme(style="whitegrid", context="talk", palette="muted")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # MAE plot
    sns.lineplot(data=df, x="epoch", y="train_mae", ax=axes[0], label="Train", linewidth=2.5)
    sns.lineplot(data=df, x="epoch", y="val_mae", ax=axes[0], label="Val", linewidth=2.5)
    axes[0].set_title("Mean Absolute Error (MAE)", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MAE (BPM)")
    axes[0].legend()
    
    # Loss (MSE) plot
    sns.lineplot(data=df, x="epoch", y="train_loss", ax=axes[1], label="Train", linewidth=2.5)
    sns.lineplot(data=df, x="epoch", y="val_loss", ax=axes[1], label="Val", linewidth=2.5)
    axes[1].set_title("Mean Squared Error (MSE)", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(out_dir / "metrics_plot.png", dpi=300)
    plt.close()
    print(f"Plot saved to {out_dir / 'metrics_plot.png'}")


def generate_report(summary: dict, out_dir: Path):
    """Generate a detailed text report of the PPG experiment."""
    config = summary["config"]
    history = summary["history"]
    report = []
    report.append("=" * 60)
    report.append("PPG-DaLiA SLinOSS EXPERIMENT REPORT")
    report.append("=" * 60)
    report.append(f"Run Name:     {config['run_name']}")
    report.append(f"Best Val MAE: {summary['best_val_mae']:.4f}")
    report.append(f"Test MAE:     {summary['test_mae']:.4f}")
    report.append(f"Test RMSE:    {summary['test_rmse']:.4f}")
    report.append("-" * 60)
    report.append("Hyperparameters:")
    for k, v in config.items():
        if k != "run_name": report.append(f"  {k:15}: {v}")
    report.append("-" * 60)
    report.append("Final Epochs (last 5):")
    for h in history[-5:]:
        report.append(
            f"  Epoch {h['epoch']:03d}: tr_mae={h['train_mae']:.4f} tr_rmse={h['train_rmse']:.4f} "
            f"va_mae={h['val_mae']:.4f} va_rmse={h['val_rmse']:.4f}"
        )
    report.append("=" * 60)
    report_text = "\n".join(report)
    (out_dir / "report.txt").write_text(report_text)
    print(f"Report saved to {out_dir / 'report.txt'}")
    print(report_text)


def main():
    parser = argparse.ArgumentParser(description="Analyze a PPG experiment run.")
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
