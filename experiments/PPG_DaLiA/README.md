# PPG-DaLiA Heart-Rate Regression Experiments

This directory contains SLinOSS experiments for heart-rate estimation on PPG-DaLiA.

## Layout

- `config.yaml`: Nested runtime defaults and sweep search space.
- `run.py`: Single-run training entry point.
- `hyperparameters/sweep.py`: Config-driven sweep runner.
- `dataloader.py`: Dataset download, processing, caching, and split creation.
- `model.py`: `PPGRegressor` built on `SLinOSSMixer`.
- `trainer.py`: Regression training and evaluation loop.
- `analysis.py`: Summary plots and text report generation.
- `utils.py`: Config resolution, validation, logging, and optimizer setup.

## Config model

- Shared settings live under `experiment`, `data`, `training`, `model`, `backend`, and `sweep`.
- The default backend is `auto`, which matches `examples/nextchar.py` and avoids forcing CuTe on unsupported setups.
- `target_mode` is validated explicitly; use `mean`, `center`, or `sequence`.

## Usage

### Single run

```bash
./scripts/guix-run python3 experiments/PPG_DaLiA/run.py
./scripts/guix-run python3 experiments/PPG_DaLiA/run.py --run-name my_ppg_run
```

### Hyperparameter sweep

```bash
./scripts/guix-run python3 experiments/PPG_DaLiA/hyperparameters/sweep.py
```

The sweep script reads its search space from `config.yaml` instead of using a hard-coded grid.

### Analysis

```bash
./scripts/guix-run python3 experiments/PPG_DaLiA/analysis.py experiments/PPG_DaLiA/runs/<run_name>
```
