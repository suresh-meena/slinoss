# DGX-H100 image for the SLinOSS UEA hyperparameter sweep workflow.
# Build this from the slinoss repo root, for example:
#   docker build -f deploy/dgxh100/uea-hypersweep-4gpu/Dockerfile \
#     --build-arg UID=$(id -u) \
#     --build-arg GID=$(id -g) \
#     --build-arg USERNAME=$USER \
#     -t $USER/slinoss:dgxh100-uea-4gpu .
#
# This starts from an official PyTorch CUDA image so the build does not need to
# install the full torch stack itself, which is expensive and was getting killed
# during the previous build.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel

ARG UID=1000
ARG GID=1000
ARG USERNAME=dgxuser

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST=9.0

SHELL ["/bin/bash", "-lc"]

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        ninja-build; \
    rm -rf /var/lib/apt/lists/*; \
    if ! getent group "${GID}" >/dev/null; then groupadd -g "${GID}" "${USERNAME}"; fi; \
    if ! id -u "${USERNAME}" >/dev/null 2>&1; then useradd -m -u "${UID}" -g "${GID}" -s /bin/bash "${USERNAME}"; fi

WORKDIR /opt/slinoss

COPY pyproject.toml setup.py requirements.txt README.md /opt/slinoss/
COPY csrc /opt/slinoss/csrc
COPY experiments/UEA /opt/slinoss/experiments/UEA
COPY scripts /opt/slinoss/scripts
COPY slinoss /opt/slinoss/slinoss

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --no-build-isolation accelerate numpy tqdm matplotlib seaborn pyyaml && \
    python3 setup.py build_ext --inplace && \
    python -m pip install --no-build-isolation -e '.[cuda]' && \
    python - <<'PY'
import torch
from slinoss.ops.cconv1d import cconv1d_is_available, cconv1d_load_error

print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cconv1d_cuda_available", cconv1d_is_available())
print("cconv1d_load_error", cconv1d_load_error())
PY

ENV HOME=/home/${USERNAME}
USER ${USERNAME}
