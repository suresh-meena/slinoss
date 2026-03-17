# DGX-H100 image for the SLinOSS UEA hyperparameter sweep workflow.
# Build this from the slinoss repo root, for example:
#   docker build -f deploy/dgxh100/uea-hypersweep-4gpu/Dockerfile \
#     -t $USER/slinoss:dgxh100-uea-4gpu .
#
# The repo requires Python >=3.11, so this image installs Miniconda on top of
# the published CUDA 12.2 DGX-H100 base.
FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

ARG UID=1000
ARG GID=1000
ARG USERNAME=dgxuser
ARG CONDA_DIR=/opt/conda
ARG CONDA_ENV=slinoss
ARG TORCH_VERSION=2.4.1
ARG TORCHVISION_VERSION=0.19.1
ARG TORCHAUDIO_VERSION=2.4.1

ENV CONDA_DIR=${CONDA_DIR}
ENV CONDA_ENV=${CONDA_ENV}
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
        bash \
        bzip2 \
        build-essential \
        ca-certificates \
        git \
        ninja-build \
        wget; \
    rm -rf /var/lib/apt/lists/*; \
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh; \
    bash /tmp/miniconda.sh -b -p "${CONDA_DIR}"; \
    rm -f /tmp/miniconda.sh; \
    "${CONDA_DIR}/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main; \
    "${CONDA_DIR}/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r; \
    "${CONDA_DIR}/bin/conda" create -y -n "${CONDA_ENV}" python=3.11 pip; \
    "${CONDA_DIR}/bin/conda" clean -afy; \
    if ! getent group "${GID}" >/dev/null; then groupadd -g "${GID}" "${USERNAME}"; fi; \
    if ! id -u "${USERNAME}" >/dev/null 2>&1; then useradd -m -u "${UID}" -g "${GID}" -s /bin/bash "${USERNAME}"; fi

ENV PATH=${CONDA_DIR}/envs/${CONDA_ENV}/bin:${CONDA_DIR}/bin:$PATH

WORKDIR /opt/slinoss

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install \
      --index-url https://download.pytorch.org/whl/cu121 \
      "torch==${TORCH_VERSION}" \
      "torchvision==${TORCHVISION_VERSION}" \
      "torchaudio==${TORCHAUDIO_VERSION}"

COPY pyproject.toml setup.py requirements.txt README.md /opt/slinoss/
COPY csrc /opt/slinoss/csrc
COPY experiments/UEA /opt/slinoss/experiments/UEA
COPY scripts /opt/slinoss/scripts
COPY slinoss /opt/slinoss/slinoss

# Build the vendored causal-conv1d extension in-place once, then install the
# package. This matches the requested repo workflow for the latest pull.
RUN python3 setup.py build_ext --inplace && \
    \
    # The extension build imports torch from setup.py, so no-build-isolation is
    # required to make the already-installed torch visible during the editable build.
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
