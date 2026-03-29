# UEA Multivariate Classification Experiments

This directory contains SLinOSS experiments for the UEA multivariate classification benchmark.

## Layout

- `config.yaml`: Nested defaults, per-dataset overrides, and per-dataset sweep spaces.
- `run.py`: Single-run training entry point.
- `precache.py`: Prebuild cached dataset splits for the configured UEA datasets.
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

Activate the project environment first:

```bash
conda activate slinoss
```

If you want multi-GPU execution, use `accelerate launch` from the same environment.

### Single run

```bash
./scripts/guix-run python3 experiments/UEA/run.py
./scripts/guix-run python3 experiments/UEA/run.py --dataset ArticularyWordRecognition --run-name my_first_run
python experiments/UEA/run.py --dataset ArticularyWordRecognition --run-name my_first_run
accelerate launch experiments/UEA/run.py --dataset ArticularyWordRecognition --run-name my_first_run
```

### Prebuild dataset cache

```bash
./scripts/guix-run python3 experiments/UEA/precache.py
./scripts/guix-run python3 experiments/UEA/precache.py --datasets EigenWorms MotorImagery
python experiments/UEA/precache.py
python experiments/UEA/precache.py --datasets EigenWorms MotorImagery
```

### Run All Datasets

Preprocess all datasets declared in `config.yaml`:

```bash
python experiments/UEA/precache.py
```

Run a hyperparameter sweep on all datasets declared in `config.yaml`:

```bash
python experiments/UEA/hyperparameters/sweep.py --sweep-name all_datasets
accelerate launch experiments/UEA/hyperparameters/sweep.py --sweep-name all_datasets
```

Run one training job per configured dataset:

```bash
for ds in EigenWorms SelfRegulationSCP1 SelfRegulationSCP2 EthanolConcentration Heartbeat MotorImagery; do
  python experiments/UEA/run.py --dataset "$ds" --run-name "${ds}_$(date +%Y%m%d-%H%M%S)"
done
```

### Hyperparameter sweep

```bash
./scripts/guix-run python3 experiments/UEA/hyperparameters/sweep.py --datasets ArticularyWordRecognition AtrialFibrillation
python experiments/UEA/hyperparameters/sweep.py --datasets EigenWorms MotorImagery
accelerate launch experiments/UEA/hyperparameters/sweep.py --datasets EigenWorms MotorImagery
```

The sweep script reads its search space from `config.yaml` instead of from hard-coded grids.

### Accelerate

Configure `accelerate` once:

```bash
accelerate config
```

Run a single dataset on multiple GPUs:

```bash
accelerate launch experiments/UEA/run.py --dataset MotorImagery --run-name motor_multi_gpu
```

Run a sweep distributed across GPUs:

```bash
accelerate launch experiments/UEA/hyperparameters/sweep.py --datasets MotorImagery --sweep-name motor_sweep_multi_gpu
```

### Analysis

```bash
./scripts/guix-run python3 experiments/UEA/analysis.py experiments/UEA/runs/ArticularyWordRecognition/<run_name>
```
