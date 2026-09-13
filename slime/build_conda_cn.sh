#!/bin/bash
# CN-mirror patched version of build_conda.sh for servers with restricted overseas access.
# Changes from original:
#   - pip uses Aliyun PyPI mirror
#   - PyTorch +cu129 wheels use Aliyun mirror
#   - sglang cloned from Gitee mirror
#   - Megatron-LM cloned from Gitee mirror
#   - apex, torch_memory_saver, FlashQLA fetched via pre-downloaded tarballs or Gitee
#   - micromamba installed from Gitee mirror
#   - conda channels use Tsinghua mirror

set -ex

export SLIME_DIR="${SLIME_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"

# Mirror settings
PIP_INDEX="https://mirrors.aliyun.com/pypi/simple/"
PIP_EXTRA="https://pypi.tuna.tsinghua.edu.cn/simple/"
TORCH_INDEX="https://mirrors.aliyun.com/pytorch-wheels/cu129/"
# Aliyun serves wheel listings as HTML directory (not PEP 503 simple index).
# Use --find-links for direct wheel resolution.
TORCH_FIND_LINKS="https://mirrors.aliyun.com/pytorch-wheels/cu129/"
# GitHub proxy (github.com is ~9 KB/s from AutoDL; route everything through gh-proxy.com)
GH="https://ghfast.top/https://github.com"
# Gitee mirrors
SGLANG_MIRROR="https://gitee.com/mirrors/sglang.git"
MEGATRON_MIRROR="https://gitee.com/mirrors/Megatron-LM.git"

WHEELS_DIR="${WHEELS_DIR:-$BASE_DIR/wheels}"

pip_install() {
  pip install "$@" -i "$PIP_INDEX" --extra-index-url "$PIP_EXTRA" \
    ${WHEELS_DIR:+--find-links "$WHEELS_DIR"}
}

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/root/micromamba}"
export MAMBA_EXE="${MAMBA_EXE:-/root/.local/bin/micromamba}"

# Use the pre-installed system conda (miniconda3 bundled with AutoDL image) to
# create the env — it has cached repodata that fits in the 2 GB container limit.
# micromamba's SAT solver OOMs at env-create time on this container.
SYSTEM_CONDA="${SYSTEM_CONDA:-/root/miniconda3/bin/conda}"
if [ ! -x "$SYSTEM_CONDA" ]; then
  echo "System conda not found at $SYSTEM_CONDA; set SYSTEM_CONDA= to override" >&2
  exit 1
fi

export PS1=tmp
mkdir -p "${CARGO_HOME:-/root/.cargo}"
touch "${CARGO_HOME:-/root/.cargo}/env"

ENV_PREFIX="$MAMBA_ROOT_PREFIX/envs/slime"
if [ ! -f "$ENV_PREFIX/conda-meta/history" ]; then
  "$SYSTEM_CONDA" create -p "$ENV_PREFIX" python=3.12 pip -y \
    --override-channels \
    -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
fi
# Activate by prepending env bin to PATH
export PATH="$ENV_PREFIX/bin:$PATH"
export CONDA_PREFIX="$ENV_PREFIX"
export CONDA_DEFAULT_ENV=slime
export CUDA_HOME="$CONDA_PREFIX"

# Always use system conda for all conda installs — micromamba OOMs at 2 GB container limit.
CONDA_CMD="$SYSTEM_CONDA"
CONDA_ARGS="-p $ENV_PREFIX"

export SGLANG_VERSION="v0.5.15.post1"
export SGLANG_COMMIT="0b3bb0cbe31873994c9f989fddfe2f87ca839fdd"
export MEGATRON_COMMIT="1dcf0dafa884ad52ffb243625717a3471643e087"
export PATCH_VERSION="v0.5.15.post1"
export TMS_COMMIT="8d30c59ca12a68d9deccbc9c6599076a1218cbc5"

export BASE_DIR=${BASE_DIR:-"/root"}
cd $BASE_DIR

# install cuda 12.9 — skip if already present (conda repodata fetch OOMs in 2GB container)
if [ ! -f "$ENV_PREFIX/lib/libcudart.so" ]; then
  $CONDA_CMD install $CONDA_ARGS \
    cuda=12.9.1 \
    cuda-nvtx=12.9.79 \
    cuda-nvtx-dev=12.9.79 \
    nccl \
    -c https://mirrors.sustech.edu.cn/anaconda-extra/cloud/nvidia/label/cuda-12.9.1 \
    -c https://mirrors.sustech.edu.cn/anaconda-extra/cloud/nvidia \
    -c conda-forge \
    -y
else
  echo "cuda already installed, skipping"
fi
if [ ! -f "$ENV_PREFIX/lib/libcudnn.so" ]; then
  $CONDA_CMD install $CONDA_ARGS -c conda-forge cudnn -y
else
  echo "cudnn already installed, skipping"
fi
# Install Rust via rustup from CN mirror instead of conda (conda rust solve OOMs in 2GB container)
export RUSTUP_DIST_SERVER=https://mirrors.ustc.edu.cn/rust-static
export RUSTUP_UPDATE_ROOT=https://mirrors.ustc.edu.cn/rust-static/rustup
if ! command -v rustc &>/dev/null; then
  curl -sSf https://mirrors.ustc.edu.cn/misc/rustup-install.sh | sh -s -- -y --no-modify-path
fi
export PATH="$HOME/.cargo/bin:$PATH"
rustc --version

# install sglang from Gitee mirror
if [ ! -d "$BASE_DIR/sglang" ]; then
  cd $BASE_DIR
  git clone "$SGLANG_MIRROR" sglang
fi
cd $BASE_DIR/sglang
git checkout ${SGLANG_COMMIT}

# Skip the entire pre-GPU block if flash-attn is already installed (resumed build).
if ! python -c "import flash_attn" 2>/dev/null; then
  # Patch sglang's overly-conservative cuda-python>=13.0 pin.
  # sglang only uses cuda.bindings.driver/runtime, which exist in 12.9.x.
  # torch+cu129 requires cuda-bindings<13, so we relax to >=12.9 to let pip resolve freely.
  sed -i 's/"cuda-python>=13\.0"/"cuda-python>=12.9"/' python/pyproject.toml
  pip install -e "python[all]" \
    --find-links "$TORCH_FIND_LINKS" \
    ${WHEELS_DIR:+--find-links "$WHEELS_DIR"} \
    -i "$PIP_INDEX" --extra-index-url "$PIP_EXTRA"
  # Force-reinstall torch again to be sure (sglang may have overwritten with cu13 variant)
  pip install --force-reinstall --no-deps \
    torch==2.11.0+cu129 torchvision==0.26.0+cu129 torchaudio==2.11.0+cu129 \
    --find-links "$TORCH_FIND_LINKS" \
    ${WHEELS_DIR:+--find-links "$WHEELS_DIR"} \
    -i "$PIP_INDEX"
  SGK_WHL="sglang_kernel-0.4.4+cu129-cp310-abi3-manylinux2014_x86_64.whl"
  if [ ! -f "$WHEELS_DIR/$SGK_WHL" ]; then
    wget -q --show-progress --retry-connrefused --tries=20 --waitretry=15 --continue \
      "$GH/sgl-project/whl/releases/download/v0.4.4/sglang_kernel-0.4.4%2Bcu129-cp310-abi3-manylinux2014_x86_64.whl" \
      -O "$WHEELS_DIR/$SGK_WHL"
  fi
  pip install --force-reinstall --no-deps "$WHEELS_DIR/$SGK_WHL"
  # sgl-deep-gemm is pure-Python (no cu suffix), already in wheels dir from pip_dl
  SGD_WHL="sgl_deep_gemm-0.1.4-py3-none-manylinux2014_x86_64.whl"
  pip install --force-reinstall --no-deps "$WHEELS_DIR/$SGD_WHL"
  pip uninstall -y \
    nvidia-cublas \
    nvidia-cuda-cupti \
    nvidia-cuda-nvrtc \
    nvidia-cuda-runtime \
    nvidia-cudnn-cu13 \
    nvidia-cufft \
    nvidia-cufile \
    nvidia-curand \
    nvidia-cusolver \
    nvidia-cusparse \
    nvidia-cusparselt-cu13 \
    nvidia-nccl-cu13 \
    nvidia-nvjitlink \
    nvidia-nvshmem-cu13 \
    nvidia-nvtx \
    nvidia-cutlass-dsl-libs-cu13 \
    || true
  pip install --force-reinstall --no-deps \
    nvidia-cublas-cu12 \
    nvidia-cuda-cupti-cu12 \
    nvidia-cuda-nvrtc-cu12 \
    nvidia-cuda-runtime-cu12 \
    "nvidia-cudnn-cu12==9.16.0.29" \
    nvidia-cufft-cu12 \
    nvidia-cufile-cu12 \
    nvidia-curand-cu12 \
    nvidia-cusolver-cu12 \
    nvidia-cusparse-cu12 \
    nvidia-cusparselt-cu12 \
    nvidia-nccl-cu12 \
    nvidia-nvjitlink-cu12 \
    nvidia-nvshmem-cu12 \
    nvidia-nvtx-cu12 \
    --find-links "$TORCH_FIND_LINKS" \
    ${WHEELS_DIR:+--find-links "$WHEELS_DIR"} \
    -i "$PIP_INDEX"

  pip_install cmake ninja

  # flash-attn: use pre-downloaded wheel if available, else download via wget (resumable)
  FLASH_ATTN_WHL="flash_attn-2.8.3+cu12torch2.11cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
  if [ -f "$WHEELS_DIR/$FLASH_ATTN_WHL" ]; then
    pip install --no-deps "$WHEELS_DIR/$FLASH_ATTN_WHL"
  elif [ -f "/tmp/$FLASH_ATTN_WHL" ]; then
    pip install --no-deps "/tmp/$FLASH_ATTN_WHL"
  else
    wget -q --show-progress --retry-connrefused --tries=20 --waitretry=15 --continue \
      "$GH/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.11/flash_attn-2.8.3%2Bcu12torch2.11cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" \
      -O "/tmp/$FLASH_ATTN_WHL"
    echo "3d0c8e60f820321eedd7166e79c33cb816263d8be6e35c3f5ba8fe2df6fea697  /tmp/$FLASH_ATTN_WHL" | sha256sum -c
    pip install --no-deps "/tmp/$FLASH_ATTN_WHL"
  fi

  pip_install flash-linear-attention==0.4.2

  # FlashQLA: try Gitee mirror first, fall back to GitHub via proxy
  pip_install git+https://gitee.com/mirrors/FlashQLA.git --no-build-isolation 2>/dev/null || \
    pip install git+$GH/QwenLM/FlashQLA.git --no-build-isolation

  # tilelang
  pip install tilelang -f https://tile-ai.github.io/whl/nightly/cu128/ -i "$PIP_INDEX"
else
  echo "flash_attn already installed, skipping pre-GPU pip block"
fi

# transformer_engine: no cu12+torch2.11 pre-built wheel exists; must build from source.
# NVTE_RELEASE_BUILD=0 prevents it from trying to fetch a pre-built wheel from GitHub.
NVTE_RELEASE_BUILD=0 NVTE_FRAMEWORK=pytorch \
  pip install --no-build-isolation --no-binary transformer_engine_torch \
  "transformer_engine[pytorch]==2.16.1" -i "$PIP_INDEX"

# apex: clone from Gitee mirror then install
if [ ! -d "$BASE_DIR/apex" ]; then
  git clone https://gitee.com/mirrors/apex.git "$BASE_DIR/apex"
fi
cd "$BASE_DIR/apex"
git checkout 10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4 2>/dev/null || {
  echo "apex commit not in Gitee mirror, fetching from GitHub..."
  git remote add upstream $GH/NVIDIA/apex.git || true
  git fetch upstream 10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4
  git checkout 10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4
}
NVCC_APPEND_FLAGS="--threads 4" \
  pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" .

TMS_CUDA_MAJOR="${TMS_CUDA_MAJOR:-$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')}"
export TMS_CUDA_MAJOR

# torch_memory_saver: clone from Gitee if available, else GitHub
if [ ! -d "$BASE_DIR/torch_memory_saver" ]; then
  git clone https://gitee.com/mirrors/torch_memory_saver.git "$BASE_DIR/torch_memory_saver" 2>/dev/null || \
    git clone $GH/zhuzilin/torch_memory_saver.git "$BASE_DIR/torch_memory_saver"
fi
cd "$BASE_DIR/torch_memory_saver"
git checkout ${TMS_COMMIT}
pip install -v . \
  --no-cache-dir --force-reinstall --no-build-isolation -i "$PIP_INDEX"

pip_install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation
SGR_WHL="sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl"
if [ -f "$WHEELS_DIR/$SGR_WHL" ]; then
  pip install "$WHEELS_DIR/$SGR_WHL" --force-reinstall -i "$PIP_INDEX"
else
  wget -q --show-progress --retry-connrefused --tries=20 --waitretry=15 --continue \
    "$GH/zhuzilin/sgl-router/releases/download/v0.3.2-9daabcd/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl" \
    -O /tmp/sglang_router.whl
  pip install /tmp/sglang_router.whl --force-reinstall -i "$PIP_INDEX"
fi
python -c "import sglang_router; assert 'slime' in sglang_router.__version__"

# Megatron from Gitee mirror
cd $BASE_DIR
if [ ! -d "$BASE_DIR/Megatron-LM" ]; then
  git clone "$MEGATRON_MIRROR" Megatron-LM --recursive
fi
pip_install "setuptools<80.0.0" pybind11 "packaging>=24.2"
cd $BASE_DIR/Megatron-LM && git checkout ${MEGATRON_COMMIT} && pip install -e . --no-build-isolation -i "$PIP_INDEX"

# install slime
cd $SLIME_DIR
pip install -r requirements.txt -i "$PIP_INDEX"
pip install -e . --no-deps

cd $SLIME_DIR/slime/backends/megatron_utils/kernels/int4_qat
pip install . --no-build-isolation -i "$PIP_INDEX"

pip_install nvidia-cudnn-cu12==9.16.0.29
pip_install "numpy==1.26.4" "scipy==1.17.1"
pip_install "kernels<0.15.0"

# apply patches
patch_dir="$SLIME_DIR/docker/patch/${PATCH_VERSION}"
if [ ! -d "$patch_dir" ]; then
  echo "Patch directory does not exist: $patch_dir" >&2
  exit 1
fi

cd $BASE_DIR/sglang
for patch_name in sglang.patch sglang-top_p.patch sglang-release_hicache.patch sglang-pull_weights.patch; do
  patch_path="$patch_dir/${patch_name}"
  if [ ! -f "$patch_path" ]; then
    continue
  fi
  if git apply --check "$patch_path"; then
    git apply "$patch_path"
  elif git apply --reverse --check "$patch_path"; then
    echo "$patch_name already applied, skipping"
  else
    echo "$patch_name does not apply cleanly" >&2
    exit 1
  fi
done
cd $BASE_DIR/Megatron-LM
megatron_patch="$patch_dir/megatron.patch"
if [ ! -f "$megatron_patch" ]; then
  echo "Megatron patch does not exist: $megatron_patch" >&2
  exit 1
fi
if git apply --reverse --check "$megatron_patch"; then
  echo "megatron.patch already applied, skipping"
else
  git update-index --refresh || true
  if ! git apply "$megatron_patch" --3way; then
    echo "megatron.patch does not apply cleanly" >&2
    exit 1
  fi
  if git grep -n '^<<<<<<< ' -- .; then
    echo "megatron patch failed to apply cleanly. Please resolve conflicts." >&2
    exit 1
  fi
fi

python - <<'PY'
import sglang
import torch
import torchaudio
import torchvision

assert torch.__version__ == "2.11.0+cu129"
assert torchaudio.__version__ == "2.11.0+cu129"
assert torchvision.__version__ == "0.26.0+cu129"
assert hasattr(torch.ops.torchvision, "nms")
PY
