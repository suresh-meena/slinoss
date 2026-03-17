#!/bin/bash
#SBATCH --job-name=uea_hypersweep2
#SBATCH --partition=q_1day-2G
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# DGX-H100 currently exposes q_1day-2G in sinfo, so this job targets the
# published 1-day 2-GPU queue directly.

set -euo pipefail

USER_NAME="${USER}"
SCRATCH="/raid/${USER_NAME}"
SCRATCH_CONT="/workspace${SCRATCH}"

REPO_NAME="${REPO_NAME:-SLinoss}"
PROJ_HOST="${PROJ_HOST:-${SCRATCH}/${REPO_NAME}/slinoss}"
PROJ_CONT="/workspace${PROJ_HOST}"
HOME_HOST="${HOME_HOST:-${SCRATCH}/.container-home/slinoss}"
HOME_CONT="/workspace${HOME_HOST}"
IMAGE="${IMAGE:-${USER_NAME}/slinoss:dgxh100-uea-4gpu}"

SWEEP_NAME="${SWEEP_NAME:-uea_hypersweep_${SLURM_JOB_ID}}"
DATASETS_STR="${DATASETS_STR:-EigenWorms SelfRegulationSCP1 SelfRegulationSCP2 EthanolConcentration Heartbeat MotorImagery}"
PRECACHE="${PRECACHE:-1}"
NUM_SEEDS="${NUM_SEEDS:-}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
MASTER_PORT="${MASTER_PORT:-$((10000 + SLURM_JOB_ID % 50000))}"
ACCELERATE_LOG_LEVEL="${ACCELERATE_LOG_LEVEL:-info}"

if [[ ! -d "${PROJ_HOST}" ]]; then
    echo "[slurm] expected project directory does not exist: ${PROJ_HOST}" >&2
    exit 1
fi

mkdir -p "${HOME_HOST}"

CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-16}"
THREADS_PER_RANK="${OMP_NUM_THREADS_OVERRIDE:-$(( (CPUS_PER_TASK + 1) / 2 ))}"
export OMP_NUM_THREADS="${THREADS_PER_RANK}"
export MKL_NUM_THREADS="${THREADS_PER_RANK}"
export NCCL_DEBUG="${NCCL_DEBUG:-warn}"
export PYTHONUNBUFFERED=1

echo "[slurm] JobID=${SLURM_JOB_ID} Node=$(hostname) GPUs=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[slurm] PROJ_HOST=${PROJ_HOST}"
echo "[slurm] IMAGE=${IMAGE}"
echo "[slurm] SWEEP_NAME=${SWEEP_NAME}"
date

docker run --rm \
  -t --gpus '"device='"${CUDA_VISIBLE_DEVICES}"'"' \
  --cpus="${CPUS_PER_TASK}" \
  --name "${SLURM_JOB_ID}" \
  --ipc=host --shm-size=64G \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --user "$(id -u "${USER_NAME}"):$(id -g "${USER_NAME}")" \
  -v "${SCRATCH}:${SCRATCH_CONT}" \
  -w "${PROJ_CONT}" \
  -e "HOME=${HOME_CONT}" \
  -e "OMP_NUM_THREADS=${OMP_NUM_THREADS}" \
  -e "MKL_NUM_THREADS=${MKL_NUM_THREADS}" \
  -e "NCCL_DEBUG=${NCCL_DEBUG}" \
  -e "PYTHONUNBUFFERED=1" \
  -e "HF_HOME=${HOME_CONT}/.cache/huggingface" \
  -e "XDG_CACHE_HOME=${HOME_CONT}/.cache" \
  -e "TORCH_HOME=${HOME_CONT}/.cache/torch" \
  -e "DATASETS_STR=${DATASETS_STR}" \
  -e "SWEEP_NAME=${SWEEP_NAME}" \
  -e "PRECACHE=${PRECACHE}" \
  -e "NUM_SEEDS=${NUM_SEEDS}" \
  -e "MIXED_PRECISION=${MIXED_PRECISION}" \
  -e "MASTER_PORT=${MASTER_PORT}" \
  -e "ACCELERATE_LOG_LEVEL=${ACCELERATE_LOG_LEVEL}" \
  "${IMAGE}" \
  bash -lc '
set -euo pipefail

mkdir -p "$HOME" "$HF_HOME" "$XDG_CACHE_HOME" "$TORCH_HOME"

echo "[container] python=$(python --version 2>&1)"
echo "[container] pwd=$(pwd)"

python - <<PY
import torch

count = torch.cuda.device_count()
print(f"[container] visible_gpus={count}")
for i in range(count):
    print(f"[container] gpu[{i}]={torch.cuda.get_device_name(i)}")
if count != 2:
    raise SystemExit(f"Expected 2 visible GPUs, got {count}.")
PY

read -r -a DATASETS <<< "$DATASETS_STR"

if [[ "${PRECACHE}" == "1" ]]; then
  if [[ "${#DATASETS[@]}" -gt 0 ]]; then
    echo "[container] precaching datasets: ${DATASETS[*]}"
    python experiments/UEA/precache.py --datasets "${DATASETS[@]}"
  else
    echo "[container] precaching datasets from config defaults"
    python experiments/UEA/precache.py
  fi
fi

launch_args=(
  --multi_gpu
  --num_processes 2
  --num_machines 1
  --mixed_precision "${MIXED_PRECISION}"
  --main_process_port "${MASTER_PORT}"
)

sweep_args=(
  experiments/UEA/hyperparameters/sweep.py
  --sweep-name "${SWEEP_NAME}"
)

if [[ "${#DATASETS[@]}" -gt 0 ]]; then
  sweep_args+=(--datasets "${DATASETS[@]}")
fi

if [[ -n "${NUM_SEEDS}" ]]; then
  sweep_args+=(--num-seeds "${NUM_SEEDS}")
fi

echo "[container] accelerate launch ${launch_args[*]} ${sweep_args[*]}"
ACCELERATE_LOG_LEVEL="${ACCELERATE_LOG_LEVEL}" accelerate launch "${launch_args[@]}" "${sweep_args[@]}" &
LAUNCH_PID=$!

trap "echo \"[container] got SIGUSR1 -> forwarding\"; kill -USR1 -${LAUNCH_PID} 2>/dev/null || true" USR1
trap "echo \"[container] got SIGTERM -> forwarding\"; kill -TERM -${LAUNCH_PID} 2>/dev/null || true" TERM

wait "${LAUNCH_PID}"
echo "[container] sweep complete"
'
