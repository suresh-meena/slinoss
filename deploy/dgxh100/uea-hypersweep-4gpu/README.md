# DGX-H100 UEA Hypersweep

This folder contains a dedicated Docker build recipe and a 4-GPU SLURM request
for `experiments/UEA/hyperparameters/sweep.py`. It does not modify any of the
existing project files.

## Build the image

Run this from the `slinoss/` repo root on DGX-H100:

```bash
docker build \
  -f deploy/dgxh100/uea-hypersweep-4gpu/Dockerfile \
  --build-arg UID=$(id -u) \
  --build-arg GID=$(id -g) \
  --build-arg USERNAME=$USER \
  -t $USER/slinoss:dgxh100-uea-4gpu .
```

## Submit the job

First confirm whether your DGX-H100 account actually exposes a 4-GPU partition:

```bash
sinfo
```

Then submit with that partition name:

```bash
sbatch --partition <4gpu_partition_name> \
  deploy/dgxh100/uea-hypersweep-4gpu/job.sbatch
```

## Useful overrides

You can override runtime settings without editing the job file:

```bash
sbatch \
  --partition <4gpu_partition_name> \
  --export=ALL,IMAGE=$USER/slinoss:dgxh100-uea-4gpu,SWEEP_NAME=uea_full_sweep,NUM_SEEDS=2 \
  deploy/dgxh100/uea-hypersweep-4gpu/job.sbatch
```

Relevant environment variables:

- `IMAGE`: Docker image tag. Default: `$USER/slinoss:dgxh100-uea-4gpu`
- `PROJ_HOST`: Host path to the `slinoss/` repo root. Default: `/raid/$USER/SLinoss/slinoss`
- `SWEEP_NAME`: Sweep output folder name
- `DATASETS_STR`: Space-separated dataset list
- `NUM_SEEDS`: Optional cap on seeds per hyperparameter combo
- `PRECACHE`: `1` to pre-download/precache UEA datasets before launching the sweep

## Queue caveat

The SERC DGX-H100 queue notes copied into this repo were last updated on
2026-03-08 and listed only 1-GPU and 2-GPU partitions. If `sinfo` still does
not show a 4-GPU queue for your account, this job cannot be scheduled as-is on
the published DGX-H100 queues.
