#!/bin/bash
# Pre-download all large wheels needed by build_conda_cn.sh into $WHEELS_DIR.
# Uses wget --continue so interrupted downloads resume from where they left off.
# Run this ONCE before build_conda_cn.sh. Subsequent build restarts use local wheels.
#
# Usage:
#   BASE_DIR=/root/autodl-tmp bash slime/download_wheels_cn.sh

set -e

BASE_DIR="${BASE_DIR:-/root/autodl-tmp}"
WHEELS_DIR="${WHEELS_DIR:-$BASE_DIR/wheels}"
PIP_INDEX="https://mirrors.aliyun.com/pypi/simple/"
PIP_EXTRA="https://pypi.tuna.tsinghua.edu.cn/simple/"
TORCH_FIND_LINKS="https://mirrors.aliyun.com/pytorch-wheels/cu129/"
GH="https://gh-proxy.com/https://github.com"

mkdir -p "$WHEELS_DIR"
cd "$WHEELS_DIR"

# wget with resume support
dl() {
  local url="$1"
  local out="$2"
  if [ -f "$out" ]; then
    echo "already exists: $out, skipping"
    return 0
  fi
  echo "downloading: $out"
  wget -q --show-progress --retry-connrefused --tries=20 --waitretry=15 \
    --continue -O "$out.part" "$url" && mv "$out.part" "$out"
}

echo "=== Downloading torch+cu129 wheels from Aliyun ==="
dl "https://mirrors.aliyun.com/pytorch-wheels/cu129/torch-2.11.0+cu129-cp312-cp312-manylinux_2_28_x86_64.whl" \
   "torch-2.11.0+cu129-cp312-cp312-manylinux_2_28_x86_64.whl"
dl "https://mirrors.aliyun.com/pytorch-wheels/cu129/torchvision-0.26.0+cu129-cp312-cp312-manylinux_2_28_x86_64.whl" \
   "torchvision-0.26.0+cu129-cp312-cp312-manylinux_2_28_x86_64.whl"
dl "https://mirrors.aliyun.com/pytorch-wheels/cu129/torchaudio-2.11.0+cu129-cp312-cp312-manylinux_2_28_x86_64.whl" \
   "torchaudio-2.11.0+cu129-cp312-cp312-manylinux_2_28_x86_64.whl"

echo "=== Downloading large sglang[all] deps via pip download ==="
# pip download fetches into the dir and skips already-present files.
# We chunk by package to make retries granular.
PIP="$(which pip)"

pip_dl() {
  "$PIP" download "$@" \
    --find-links "$TORCH_FIND_LINKS" \
    --find-links "$WHEELS_DIR" \
    -i "$PIP_INDEX" \
    --extra-index-url "$PIP_EXTRA" \
    -d "$WHEELS_DIR" \
    --no-cache-dir
}

# These are the known large wheels — download individually so each is resumable
# by simply re-running the script (pip download skips already-downloaded files).
pip_dl "flashinfer_cubin==0.6.12"
pip_dl "sglang-kernel==0.4.4" --extra-index-url https://docs.sglang.ai/whl/cu129/
pip_dl "sgl-deep-gemm==0.1.4" --extra-index-url https://docs.sglang.ai/whl/cu129/
pip_dl "nvidia-cudnn-cu12==9.17.1.4"
pip_dl "nvidia-cublas-cu12==12.9.1.4.*"
pip_dl "nvidia-cusparselt-cu12==0.7.1"
pip_dl "nvidia-nccl-cu12==2.28.9"
pip_dl "nvidia-nvshmem-cu12==3.4.5"
pip_dl "triton==3.6.0"
pip_dl "nvidia-cufft-cu12==11.4.1.4.*"
pip_dl "nvidia-cusolver-cu12==11.7.5.82.*"
pip_dl "nvidia-cusparse-cu12==12.5.7.53.*"
pip_dl "nvidia-nvjitlink-cu12==12.9.86.*"
pip_dl "nvidia-cuda-nvrtc-cu12==12.9.86.*"
pip_dl "nvidia-cudnn-cu12==9.16.0.29"

echo "=== Downloading flash-attn wheel via wget (resumable) ==="
dl "$GH/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.11/flash_attn-2.8.3%2Bcu12torch2.11cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" \
   "flash_attn-2.8.3+cu12torch2.11cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

echo "=== Downloading sgl-router wheel via wget (resumable) ==="
dl "$GH/zhuzilin/sgl-router/releases/download/v0.3.2-9daabcd/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl" \
   "sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl"

echo ""
echo "=== All wheels downloaded to $WHEELS_DIR ==="
ls -lh "$WHEELS_DIR" | sort -k5 -rh | head -30
du -sh "$WHEELS_DIR"
