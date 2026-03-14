# UEA Multivariate Classification Experiments

This directory contains SLinOSS experiments for the UEA multivariate classification benchmark.

## Layout

- `config.yaml`: Nested defaults, per-dataset overrides, and per-dataset sweep spaces.
- `run.py`: Single-run training entry point.
- `hyperparameters/sweep.py`: Config-driven sweep runner.
- `dataloader.py`: UEA download, parsing, caching, and split creation.
- `model.py`: `UEAClassifier` built on `SLinOSSMixer`.
- `trainer.py`: Training and evaluation loop.
- `analysis.py`: Summary plots and text report generation.
- `utils.py`: Config resolution, validation, logging, and optimizer setup.

## Config model

- Shared settings live under `experiment`, `data`, `training`, `model`, `backend`, and `sweep`.
- Dataset-specific overrides live under `datasets.<dataset_name>`.
- Dataset-specific sweep grids live under `datasets.<dataset_name>.sweep.grid`.
- The default backend is `auto`, which matches `examples/nextchar.py` and avoids forcing CuTe on unsupported setups.

## Usage

### Single run

```bash
./scripts/guix-run python3 experiments/UEA/run.py
./scripts/guix-run python3 experiments/UEA/run.py --dataset ArticularyWordRecognition --run-name my_first_run
```

### Hyperparameter sweep

```bash
./scripts/guix-run python3 experiments/UEA/hyperparameters/sweep.py --datasets ArticularyWordRecognition AtrialFibrillation
```

The sweep script reads its search space from `config.yaml` instead of from hard-coded grids.

### Analysis

```bash
./scripts/guix-run python3 experiments/UEA/analysis.py experiments/UEA/runs/ArticularyWordRecognition/<run_name>
```
