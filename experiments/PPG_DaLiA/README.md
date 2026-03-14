# PPG-DaLiA Heart-Rate Regression Experiments

This directory contains experiments for training the SLinOSS mixer on the PPG-DaLiA dataset for heart-rate estimation.

## New Structure

- `run.py`: Main entry point for training.
- `config.yaml`: Central configuration for hyperparameters and dataset settings.
- `dataloader.py`: Optimized data loading with caching of processed splits.
- `model.py`: Modular implementation of the `PPGRegressor` using SLinOSS.
- `trainer.py`: Encapsulated training and evaluation logic for regression.
- `analysis.py`: Script to generate detailed reports (`report.txt`) and plots (`metrics_plot.png`).
- `utils.py`: Logging, seeding, and configuration utilities.
- `hyperparameters/sweep.py`: Grid search utility for hyperparameter optimization.

## Usage

### Single Run

To run an experiment with the default configuration:
```bash
python run.py
```

To specify a custom run name:
```bash
python run.py --run-name my_ppg_run
```

### Hyperparameter Sweep

To run a grid search:
```bash
python hyperparameters/sweep.py
```

### Analysis

Analyze a completed run:
```bash
python analysis.py runs/<run_name>
```

## Features

- **CuTe Backend**: Fully supports the high-performance CuTe scan kernel.
- **Efficient Loading**: Processes all subjects, performs windowing, and caches the resulting splits.
- **Regression Optimized**: Focuses on MAE (BPM) and RMSE metrics.
- **Beautiful Plots**: Automatic generation of MAE and MSE curves.
