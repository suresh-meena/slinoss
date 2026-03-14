# UEA Multivariate Classification Experiments

This directory contains experiments for training the SLinOSS mixer on the UEA multivariate time series classification benchmark.

## New Structure

- `run.py`: Main entry point for training.
- `config.yaml`: Central configuration for hyperparameters and dataset settings.
- `dataloader.py`: Optimized data loading with caching to speed up repeated runs.
- `model.py`: Modular implementation of the `UEAClassifier` using SLinOSS.
- `trainer.py`: Encapsulated training and evaluation logic.
- `analysis.py`: Script to generate detailed reports (`report.txt`) and plots (`metrics_plot.png`).
- `utils.py`: Logging, seeding, and configuration utilities.
- `hyperparameters/sweep.py`: Grid search utility for hyperparameter optimization.

## Usage

### Single Run

To run an experiment with the default configuration:
```bash
python run.py
```

To override the dataset or specify a run name:
```bash
python run.py --dataset ArticularyWordRecognition --run-name my_first_run
```

### Hyperparameter Sweep

To run a grid search over predefined hyperparameters:
```bash
python hyperparameters/sweep.py --datasets ArticularyWordRecognition AtrialFibrillation
```

### Analysis

After a run is complete, analyze the results and generate plots:
```bash
python analysis.py runs/ArticularyWordRecognition/<run_name>
```

## Features

- **CuTe Backend**: Uses the high-performance CuTe scan kernel on NVIDIA GPUs.
- **Optimized Loading**: First-time runs parse `.ts` files; subsequent runs load from a fast binary cache.
- **Beautiful Plots**: Automatic generation of high-quality loss and accuracy curves.
- **Detailed Logging**: Comprehensive logs saved to each run directory.
