# Pi-Slime-ShopSimulator Project

## Workflow: Local ↔ Remote

- **We work locally** on `/home/isaber/pi-slime-shopsimulator/`
- **All file edits are made locally**, then committed and pushed to GitHub via `git`
- **The server pulls** from GitHub to get changes — never edit files directly on the server
- File transfer chain: **local edit → `git commit` → `git push origin main` → `ssh autodl "git pull"` on server**
- The git remote is `origin` at `git@github-saber:IiCEight/pi-slime-shopsimulator.git`
- To push: you need the SSH agent running — run `eval "$(ssh-agent -s)" && ssh-add ~/.ssh/id_ed25519_saber` first if `git push` fails with "does not appear to be a git repository"
- **Server git pull is slow/fails** (GitHub ~9 KB/s from AutoDL) — remote URL on server is set to `https://ghfast.top/https://github.com/IiCEight/pi-slime-shopsimulator.git`; if it breaks, re-run: `git remote set-url origin https://ghfast.top/https://github.com/IiCEight/pi-slime-shopsimulator.git`

## Server Access

- SSH alias: `ssh autodl` (configured in `~/.ssh/config`)
- Server: `connect.westb.seetacloud.com:42698`, user `root`, key `id_ed25519_saber`
- AutoDL rented Pro 6000D (84 GB VRAM), CUDA driver 595.58.03
## Disk Layout on Server

| Path | Size | Notes |
|---|---|---|
| `/` (overlay) | 30 GB | Container root — keep free, don't install big things here |
| `/root/autodl-tmp` | **50 GB** | Persistent data disk — all build work goes here |
| `/root/autodl-pub` | 20 TB | Public shared data (read-only) |

**Important:** The system `/root/miniconda3` (17 GB) and `/root/.cache` (pip cache) were on `/`. To save space:
- pip cache symlinked: `/root/.cache/pip` → `/root/autodl-tmp/.cache/pip`
- rustup symlinked: `/root/.rustup` → `/root/autodl-tmp/.rustup`
- All build output (conda env, sglang, Megatron-LM, etc.) lives under `/root/autodl-tmp/`

- Build log: `/root/autodl-tmp/build.log`
- Build session: tmux `build` — check with `ssh autodl "tail -30 /root/autodl-tmp/build.log"`
## Two-Phase Build Strategy

### Phase 1 — Pre-download wheels (run once, fully resumable)
```bash
ssh autodl "tmux new-session -d -s dl 2>/dev/null || true && tmux send-keys -t dl 'BASE_DIR=/root/autodl-tmp bash /root/autodl-tmp/pi-slime-shopsimulator/slime/download_wheels_cn.sh 2>&1 | tee /root/autodl-tmp/download.log' Enter"
```
- Uses `wget --continue` for large wheels (flash-attn, sgl-router)
- Uses `pip download` (skips already-present files) for PyPI wheels
- All wheels saved to `/root/autodl-tmp/wheels/`
- **Safe to interrupt and re-run** — resumes from where it left off

### Phase 2 — Build (uses local wheels, no re-downloading)
```bash
ssh autodl "tmux new-session -d -s build 2>/dev/null || true && tmux send-keys -t build 'export SLIME_DIR=/root/autodl-tmp/pi-slime-shopsimulator/slime BASE_DIR=/root/autodl-tmp MAMBA_ROOT_PREFIX=/root/autodl-tmp/micromamba MAMBA_EXE=/root/.local/bin/micromamba SYSTEM_CONDA=/root/miniconda3/bin/conda WHEELS_DIR=/root/autodl-tmp/wheels && bash \$SLIME_DIR/build_conda_cn.sh 2>&1 | tee /root/autodl-tmp/build.log' Enter"
```
- `build_conda_cn.sh` uses `--find-links /root/autodl-tmp/wheels/` for all installs
- Restarts are fast — wheels are already on disk



## Server Constraints

- **2 GB cgroup memory limit** — micromamba SAT solver OOMs → use system conda `/root/miniconda3/bin/conda` for all conda ops
- **GitHub nearly blocked** (~9 KB/s) → use CN mirrors for everything
- **PyTorch Aliyun mirror** serves HTML directory (not PEP 503) → use `--find-links`, not `--index-url`

## CN Mirror Strategy (in build_conda_cn.sh)

| Resource | Mirror |
|---|---|
| pip packages | `https://mirrors.aliyun.com/pypi/simple/` |
| pip extra | `https://pypi.tuna.tsinghua.edu.cn/simple/` |
| torch +cu129 wheels | `https://mirrors.aliyun.com/pytorch-wheels/cu129/` (use `--find-links`) |
| conda channels | SUSTech nvidia, conda-forge |
| sglang git | `https://gitee.com/mirrors/sglang.git` |
| Megatron-LM git | `https://gitee.com/mirrors/Megatron-LM.git` |
| rustup | `https://mirrors.ustc.edu.cn/rust-static` (USTC mirror) |
| Node.js | direct binary from `https://npmmirror.com/mirrors/node/` |

## Build Script

`slime/build_conda_cn.sh` — CN-mirror patched version of `slime/build_conda.sh`

Key fixes applied over original:
1. Use system conda (not micromamba) to avoid 2 GB OOM at env creation
2. Use `--find-links` for torch +cu129 wheels (Aliyun is HTML dir, not PEP 503)
3. Install `torch==2.11.0+cu129` with `--no-deps` BEFORE `sglang[all]` to prevent pip pulling cu13 variants
4. Pin `cuda-python==12.9` and `torch==2.11.0+cu129` explicitly in the `sglang[all]` install to prevent pip backtracking loop (cuda-python>=13 conflicts with torch+cu129's cuda-bindings<13)
5. Use Gitee mirrors for sglang, Megatron-LM, apex, torch_memory_saver
6. Use USTC rustup mirror instead of conda rust (conda rust solve OOMs)
7. Activate env via `export PATH="$ENV_PREFIX/bin:$PATH"` instead of shell hook (hook kills headless SSH session)

## Build Progress (as of 2026-09-08)

### Completed ✅
- slime conda env at `/root/autodl-tmp/micromamba/envs/slime` (Python 3.12)
- CUDA 12.9.1 + cuDNN via conda (SUSTech mirror)
- Rust 1.98.1 via USTC rustup
- `cuda-python==12.9`
- sglang cloned from Gitee @ commit `0b3bb0cbe3`
- `torch==2.11.0+cu129` + torchvision + torchaudio (Aliyun --find-links)

### In Progress 🔄
- `pip install -e "python[all]" "cuda-python==12.9" "torch==2.11.0+cu129" ...` (sglang[all] install)
- Build is running in tmux `build` on the server

### Still Pending ⏳
- Force-reinstall torch+cu129 (post-sglang cleanup)
- sglang-kernel==0.4.4 + sgl-deep-gemm==0.1.4 (cu129 index)
- Uninstall nvidia-cu13 libs, reinstall nvidia-cu12
- flash-attn pinned wheel (GitHub releases — may be slow)
- flash-linear-attention, FlashQLA, tilelang
- transformer_engine==2.16.1
- apex (Gitee clone + build from source)
- torch_memory_saver
- nvidia-modelopt, sgl-router
- Megatron-LM (Gitee clone @ commit `1dcf0dafa884`)
- slime itself (`pip install -r requirements.txt && pip install -e .`)
- int4_qat kernel
- patches (sglang.patch, megatron.patch, etc.)
- Final assertion check

## Version Pins (repro.lock.json)

| Component | Version |
|---|---|
| torch | 2.11.0+cu129 |
| sglang | 0.5.15.post1 @ `0b3bb0cbe3` |
| Megatron-LM | 0.16.0rc0 @ `1dcf0dafa884` |
| Pi agent | @earendil-works/pi-coding-agent@0.84.2 |
| Python | 3.12 |
| CUDA | 12.9.1 |

## Git Config on Server

```
user.name = iSaber
user.email = 1346959878@qq.com
```
