# Pi + Slime + ShopSimulator

本目录提供 Qwen3.5-2B + Pi + Slime + ShopSimulator 的最小训练参考实现，覆盖 512-task 教师数据采集、SFT、在线 GRPO 和单次 rollout 评测。当前版本不锁定数据、模型、环境或产物 SHA，也不承诺与我们的历史运行逐项一致。

## 历史实验结果（仅供参考）

以下结果来自精简发布目录之前的一次内部运行。它们用于展示这条训练路线在当时软硬件、依赖版本、数据采样和 checkpoint 下的效果，不是当前代码的复现承诺。

对应的 Hugging Face 发布产物：

- SFT 数据集（私有）：[`mrzhao13/pi-slime-shopsimulator-sft-512`](https://huggingface.co/datasets/mrzhao13/pi-slime-shopsimulator-sft-512)
- SFT 模型（公开）：[`mrzhao13/qwen3.5-2b-shopsimulator-sft-512-1ep`](https://huggingface.co/mrzhao13/qwen3.5-2b-shopsimulator-sft-512-1ep)
- GRPO/RL 模型（公开）：[`mrzhao13/qwen3.5-2b-shopsimulator-grpo-rl500-1ep`](https://huggingface.co/mrzhao13/qwen3.5-2b-shopsimulator-grpo-rl500-1ep)

私有 SFT 数据集仅用于归档历史运行中通过筛选的教师数据，不是运行本仓库流程的前置依赖；实验 1 会从内置 `sft_512` 任务重新采集和准备 SFT 数据。

| 阶段 | 当时的运行结果 |
| --- | --- |
| 512-task 教师采集 | 512 个任务各采集 1 条；412 条轨迹通过筛选，task 覆盖率 80.5%；转换得到 6153 个 turn-level SFT 样本。 |
| SFT | 6153 个样本完整训练 1 epoch，共 2051 个 optimizer step；生成 HF 与 Megatron checkpoint。 |
| 在线 GRPO | `rl_500` 训练 1 epoch：500 个 group、2000 个 candidate、100 个 rollout/optimizer step；352 个 group 具有非零 reward 方差。 |
| 最终评测 | 在 `official_test_200` 上对 Base、SFT、RL 各做 200 次单样本 rollout；结果见下表。 |

| 模型 | 正奖励 pass@1 | 严格成功 pass@1 | mean@1 `r_loose` | mean@1 `r_hard` |
| --- | ---: | ---: | ---: | ---: |
| Base | 2.0% | 0.0% | 0.004286 | 0.000000 |
| SFT | 72.5% | 10.5% | 0.389829 | 0.124417 |
| RL | 90.5% | 31.0% | 0.627786 | 0.354530 |

精简后的示例会按实际输入行数运行，只保留必要的数据格式、loss mask、rollout 分组和 reward 一致性检查；固定 revision、工作树状态、manifest 身份、数据/checkpoint SHA 以及历史产物完整性检查均不再作为启动条件。

## ShopSimulator 相对基线的补丁

本仓库不复制或重新分发 ShopSimulator 源码，只提供一个相对于指定上游提交的补丁，用于构建与 Slime 集成的环境。

- 上游仓库：<https://github.com/ShopAgent-Team/ShopSimulator>
- 固定补丁基线：`51bb26012cee31aea7ac26177c5ffe807026ac07`
- 补丁对应的测试版本：`3ab366b2982e9ffa59957086d0845f955ef2245b`
- 补丁文件：[`shopsimulator_patch/shopsimulator-slime-integration.patch`](shopsimulator_patch/shopsimulator-slime-integration.patch)
- SHA-256：`c75973456391f8f7784e0823bde2eae5cd6f4ae519dae51d43ed5c8b86c19dac`

补丁是上述两个提交之间的完整 Git diff，包含 50 个文件的变化。除功能修改外，它还删除了上游提交中误跟踪的 Python 缓存和运行日志，因此体积约为 382 KiB。

### 并发 rollout 与会话隔离

- 将 Flask API 改为线程化服务，并为环境池和单个环境增加锁。
- 启动时预创建 20 个环境；这些环境共享只读的 `SimServer` 数据，避免重复加载商品和目标数据。
- 为每条 rollout 引入独立的 `rollout_session_id`，将任务索引 `idx` 与会话标识分离。
- `reset` 分配空闲环境，`interact` 校验环境和会话是否匹配，任务结束后自动释放环境和会话状态。
- 完善 `release_one`、`release_all`，并增加 `status`，可查看容量、空闲环境和活动会话。
- 环境分配后发生异常时自动回收资源，避免环境池泄漏。

这些修改用于防止 Slime 并发采样时不同 rollout 共享或覆盖交互状态。

### 可复现的价格与价格约束

- 商品价格不再依赖进程级全局随机状态，而是根据商品 ASIN 确定性生成。
- 目标中的 `price_upper` 根据 ASIN 和 instruction 确定性生成。
- 同一任务在不同服务进程、环境实例和多次启动中使用相同价格数据，避免 `r_price` 与 `r_hard` 因服务端随机性漂移。

### 文本环境适配

- 修正自定义字符串会话下任务索引的传递，确保目标仍由请求中的 `idx` 选择。
- 增加单会话状态释放接口，只清理该 rollout 的可变状态。
- 文本 API 的空图像占位改用 NumPy，去除该路径对 Torch 的非必要依赖。

### 运行依赖与仓库清理

- 新增 `shop_env/requirements.runtime.txt`，记录本项目使用的 Python 运行依赖版本。
- 新增 `.gitignore`，忽略 Python 缓存、日志、PID 和本地编辑器状态。
- 从版本控制快照中删除已提交的 `__pycache__`、`.pyc` 和 `shop_agent.log` 等运行产物。

### 补丁校验与使用

完整的 ShopSimulator 获取、基线检出、补丁校验/应用、运行依赖安装和服务启动命令只在后文“[从 clone 开始运行四个实验](#从-clone-开始运行四个实验)”的第 2 步维护。应用前请同时核对上述基线 revision 和补丁 SHA-256；不要在其他 ShopSimulator 版本上强制应用。

### 许可边界

该补丁只描述本项目对指定 ShopSimulator 快照所做的差异，不包含完整上游源码，也不替代或变更上游项目的许可条款。使用者应自行查看上游仓库当前的许可与使用条件，并确保其获取、使用和分发行为拥有相应授权。

## Slime 相对固定官方基线的修改

本节固定以本项目当时 fork Slime 时使用的官方提交为比较基准，不跟随 Slime 上游后续更新：

- 官方仓库：<https://github.com/THUDM/slime>
- 固定官方基线：`624b824a898ab0ec1fcb4d373004c7f3852bf515`（2026-08-21，`[NFC] Add observability subfolder (#2298)`）
- 本仓库中的修改后源码：[`slime/`](slime/)

后续即使官方仓库发生更新，也不应在未重新审查兼容性和冲突的情况下替换上述基线。下面列出的内容均指当前 `slime/` 相对该固定提交的修改。

### 新增 ShopSimulator 实验示例

固定官方基线中没有 `examples/ShopSimulator/`。本项目新增了该目录及以下能力：

- 新增 `pi_harness.py`、`shop_extension.ts`、`generate.py` 和 `common.py`，将 Pi 多轮工具调用、ShopSimulator HTTP API、Slime rollout 和 reward 计算连接起来。
- 新增 `collect_sft.py` 与 `prepare_sft.py`，用于采集教师轨迹并转换为 turn-level SFT 数据。
- 新增 `run_sft.sh`，提供 Qwen3.5-2B 的 SFT 训练入口。
- 新增 `run_rl.sh` 与 `config/shop_rl.json`，提供基于 ShopSimulator 在线 rollout 的 GRPO 训练入口。
- 新增 `run_eval.sh`、`config/shop_eval_official_k1.yaml` 与 `summarize_eval.py`，提供单次 rollout 评测及指标汇总入口。
- 新增 `data/tasks_v2/` 下的 `sft`、`rl`、`dev`、`official_test` 四个任务池，以及实验入口使用的 `sft_512`、`rl_500` 和 `official_test_200` 数据文件。

### Qwen3.5-2B 与 checkpoint 兼容

- 新增 `scripts/models/qwen3.5-2B.sh`，补充 Qwen3.5-2B 在 Megatron 中使用的模型结构参数。
- 调整 `tools/convert_hf_to_torch_dist.py`：仅在 Megatron 参数解析器尚未注册时添加 `--use-gated-attention` 和 `--padded-vocab-size`，避免新版本 Megatron 因重复参数定义退出，同时保持旧版本兼容。

### Qwen3.5 loss mask 与空 think 块

- 修正 `slime/utils/mask_utils.py` 对 Qwen3.5 chat template 的处理。
- thinking 关闭时，模板注入的完整空块 `<think>\n\n</think>\n\n` 被视为 prompt，不参与 loss。
- thinking 开启时，只屏蔽模板注入的 `<think>\n` 前缀；模型生成的 reasoning 内容继续参与训练。
- 该修改不改变 token 序列，只修正训练 mask 的起点，并继续检查文本 tokenization 与 `apply_chat_template(..., tokenize=True)` 的结果一致。

### 多轮 adapter 的终止状态

- 在 `slime/agent/adapters/common.py` 中暴露每个 session 已完成并写入 trajectory 的真实模型 turn 数。
- 为 turn 上限增加结构化的 `turn_limit` 终止原因，供 ShopSimulator rollout 区分正常达到上限与其他 429 或基础设施异常。
- session 打开、完成或丢弃时清理 turn 计数与终止原因，避免复用 session id 时残留旧状态。

### 安装与测试兼容

- 调整 `build_conda.sh`，支持在无交互 AutoDL 会话中直接初始化或复用 micromamba，避免依赖 shell 启动脚本，并可复用已存在的 `slime` 环境。
- 更新 adapter 测试，验证真实 turn 数、`turn_limit` 原因及 session 清理。
- CPU-only agent rollout 测试仅在本机确实没有安装 `transformers` 时注入 stub，避免覆盖已经可用的真实包。

## 从 clone 开始运行四个实验

下面的命令按“512-task 教师数据采集 → 全量 SFT → `rl_500` GRPO → `official_test_200` 单次 rollout 评测”的顺序执行。

当前启动配置面向单机单卡 NVIDIA GPU；历史运行使用 84 GB 显存的 Pro 6000D，训练侧 `MAX_TOKENS_PER_GPU=12288`。更小显存配置需要重新调整 token budget、batch size 和 SGLang 显存比例，本仓库尚未验证。

### 0. 克隆仓库并定义路径

```bash
git clone https://github.com/Piucente/pi-slime-shopsimulator.git
cd pi-slime-shopsimulator

export PROJECT_ROOT="$PWD"
export THIRD_PARTY_ROOT=/absolute/path/to/pi-slime-work
export SLIME_DIR="$PROJECT_ROOT/slime"
export MAMBA_ROOT_PREFIX="$THIRD_PARTY_ROOT/micromamba"
export MAMBA_EXE=/root/.local/bin/micromamba
export BASE_DIR="$THIRD_PARTY_ROOT"

mkdir -p "$THIRD_PARTY_ROOT"
```

`THIRD_PARTY_ROOT` 用于放置 micromamba、SGLang、Megatron-LM 和 ShopSimulator；不要把这些运行环境目录提交到本仓库。非 root 用户应把 `MAMBA_EXE` 改为自己的 micromamba 安装位置。

### 1. 安装 Pi 与 Slime 训练环境

先准备 Node.js `>=22.19.0`，再安装本项目使用的 Pi 版本：

```bash
node --version
npm --version
npm install --global @earendil-works/pi-coding-agent@0.84.2

export PI_BIN="$(command -v pi)"
"$PI_BIN" --version
```

然后运行修改后的 Slime 安装脚本。它会创建或复用名为 `slime` 的 micromamba 环境，在 `THIRD_PARTY_ROOT` 下检出脚本固定的 SGLang 与 Megatron-LM revision，并安装 CUDA 12.9、PyTorch 2.11 和相应依赖：

```bash
export SLIME_DIR="$PROJECT_ROOT/slime"
export BASE_DIR="$THIRD_PARTY_ROOT"
export MAMBA_ROOT_PREFIX="$THIRD_PARTY_ROOT/micromamba"
export MAMBA_EXE=/root/.local/bin/micromamba

bash "$SLIME_DIR/build_conda.sh"

export SLIME_PYTHON="$MAMBA_ROOT_PREFIX/envs/slime/bin/python"
export MEGATRON_DIR="$THIRD_PARTY_ROOT/Megatron-LM"

"$SLIME_PYTHON" -c 'import ray, sglang, torch; print(torch.__version__, torch.version.cuda)'
```

`build_conda.sh` 会下载依赖、编译 CUDA 扩展并修改它检出的 SGLang/Megatron-LM 工作树，耗时较长。后续命令都应继续使用这里的 `SLIME_PYTHON` 和 `MEGATRON_DIR`。

### 2. 获取、打补丁并启动 ShopSimulator

```bash
export SHOP_SIM_DIR="$THIRD_PARTY_ROOT/ShopSimulator"

git clone https://github.com/ShopAgent-Team/ShopSimulator.git "$SHOP_SIM_DIR"
git -C "$SHOP_SIM_DIR" checkout 51bb26012cee31aea7ac26177c5ffe807026ac07
git -C "$SHOP_SIM_DIR" apply --check "$PROJECT_ROOT/shopsimulator_patch/shopsimulator-slime-integration.patch"
git -C "$SHOP_SIM_DIR" apply "$PROJECT_ROOT/shopsimulator_patch/shopsimulator-slime-integration.patch"

"$SLIME_PYTHON" -m pip install -r "$SHOP_SIM_DIR/shop_env/requirements.runtime.txt"
```

在单独终端启动服务，并在教师采集、GRPO 和评测期间保持运行：

```bash
cd "$SHOP_SIM_DIR/shop_env"
"$SLIME_PYTHON" shop_env/pack_api.py
```

在另一个终端确认 20-slot 环境池可用：

```bash
curl -sS -X POST http://127.0.0.1:5000/api/shop_agent \
  -H 'content-type: application/json' \
  --data '{"action":"status"}'
```

### 3. 准备 Qwen3.5-2B 的两种 checkpoint

四个实验使用同一份基础模型。先下载 [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B)，再转换出训练侧使用的 Megatron `torch_dist` checkpoint：

```bash
mkdir -p "$THIRD_PARTY_ROOT/models"
export HF_CLI="$MAMBA_ROOT_PREFIX/envs/slime/bin/hf"
export BASE_HF_CHECKPOINT="$THIRD_PARTY_ROOT/models/Qwen3.5-2B"
export BASE_MEGATRON_CHECKPOINT="$THIRD_PARTY_ROOT/models/Qwen3.5-2B_torch_dist"

"$HF_CLI" download Qwen/Qwen3.5-2B --local-dir "$BASE_HF_CHECKPOINT"

cd "$SLIME_DIR"
source scripts/models/qwen3.5-2B.sh
PYTHONPATH="$MEGATRON_DIR:$SLIME_DIR" "$SLIME_PYTHON" tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" --hf-checkpoint "$BASE_HF_CHECKPOINT" --save "$BASE_MEGATRON_CHECKPOINT"
```

HF checkpoint 提供 tokenizer 和 SGLang rollout 权重，Megatron checkpoint 提供训练权重；SFT、RL 和评测启动器需要成对传入匹配的两种格式。

### 实验 1：采集 `sft_512` 教师数据

教师采集默认读取仓库中的 `sft_512.jsonl`，每个任务采集一次，并使用 DeepSeek 兼容 API。仓库提供 `deepseek_api_key.example.txt`，它只包含占位符；不要直接在示例文件中填写真实 key。先将它复制到仓库外，真实 API key 只写在新文件的第一行：

```bash
export TEACHER_API_KEY_FILE="$THIRD_PARTY_ROOT/deepseek_api_key.txt"
cp -n "$SLIME_DIR/examples/ShopSimulator/deepseek_api_key.example.txt" "$TEACHER_API_KEY_FILE"
chmod 600 "$TEACHER_API_KEY_FILE"

# 用文本编辑器把第一行替换为真实 API key。
export SFT_DATA_ROOT=/absolute/path/to/runs/shop_sft_512

cd "$SLIME_DIR"
"$SLIME_PYTHON" -m examples.ShopSimulator.collect_sft \
  --output-dir "$SFT_DATA_ROOT" \
  --dry-run

PI_BIN="$PI_BIN" "$SLIME_PYTHON" -m examples.ShopSimulator.collect_sft \
  --output-dir "$SFT_DATA_ROOT" \
  --api-key-file "$TEACHER_API_KEY_FILE" \
  --model deepseek-v4-flash \
  --base-url https://api.deepseek.com \
  --env-url http://127.0.0.1:5000/api/shop_agent \
  --samples-per-task 1 \
  --concurrency 4
```

采集器会把每条原始轨迹写入 `$SFT_DATA_ROOT/raw/`，同一路径重跑时会跳过已有结果并继续未完成任务。采集结束后，将通过筛选的轨迹转换为独立的 turn-level SFT 样本：

```bash
cd "$SLIME_DIR"
"$SLIME_PYTHON" -m examples.ShopSimulator.prepare_sft \
  --input-dir "$SFT_DATA_ROOT" \
  --output-dir "$SFT_DATA_ROOT/prepared" \
  --tokenizer "$BASE_HF_CHECKPOINT" \
  --max-tokens 16384
```

检查 `$SFT_DATA_ROOT/summary.json`、`$SFT_DATA_ROOT/prepared/turn_examples_summary.json` 和最终的 `$SFT_DATA_ROOT/prepared/turn_examples.jsonl`。实际通过数量取决于教师输出，不要求等于历史运行的 412 条轨迹和 6153 个 turn。

### 实验 2：全量 SFT 训练 1 epoch

`run_sft.sh` 会读取 `turn_examples.jsonl` 的全部非空行，`NUM_DATA_PASSES=1` 表示完整训练一遍。默认参考实验使用 `GLOBAL_BATCH_SIZE=3`；它必须整除实际样本行数，不整除时应改为样本数的其他因数。

```bash
export SFT_RUN_ROOT=/absolute/path/to/runs/qwen35_2b_shop_sft

FULL_DATA="$SFT_DATA_ROOT/prepared/turn_examples.jsonl" \
HF_CHECKPOINT="$BASE_HF_CHECKPOINT" \
REF_MODEL_PATH="$BASE_MEGATRON_CHECKPOINT" \
RUN_ROOT="$SFT_RUN_ROOT" \
SLIME_PYTHON="$SLIME_PYTHON" \
MEGATRON_DIR="$MEGATRON_DIR" \
NUM_DATA_PASSES=1 \
GLOBAL_BATCH_SIZE=3 \
MAX_TOKENS_PER_GPU=12288 \
bash "$SLIME_DIR/examples/ShopSimulator/run_sft.sh"
```

`RUN_ROOT` 必须是尚不存在的新目录。训练完成后，HF export 位于 `$SFT_RUN_ROOT/hf/`，Megatron checkpoint 根目录为 `$SFT_RUN_ROOT/checkpoints/`。先查看实际生成的 HF 子目录，再为下一步设置路径：

```bash
find "$SFT_RUN_ROOT/hf" -mindepth 1 -maxdepth 1 -type d -print

export SFT_HF_CHECKPOINT=/absolute/path/to/the/generated/sft/hf/export
export SFT_MEGATRON_CHECKPOINT="$SFT_RUN_ROOT/checkpoints"
```

### 实验 3：使用 `rl_500` 训练 GRPO 1 epoch

默认 `config/shop_rl.json` 已配置 500 个任务、每题 4 个 candidate、rollout batch size 5 和 1 epoch。RL 必须从上一步相互匹配的 SFT HF/Megatron checkpoint 启动。

```bash
export RL_RUN_ROOT=/absolute/path/to/runs/qwen35_2b_shop_rl

HF_CHECKPOINT="$SFT_HF_CHECKPOINT" \
REF_MODEL_PATH="$SFT_MEGATRON_CHECKPOINT" \
RUN_ROOT="$RL_RUN_ROOT" \
SLIME_PYTHON="$SLIME_PYTHON" \
MEGATRON_DIR="$MEGATRON_DIR" \
PI_BIN="$PI_BIN" \
SHOP_ENV_URL=http://127.0.0.1:5000/api/shop_agent \
MAX_TOKENS_PER_GPU=12288 \
SGLANG_MEM_FRACTION_STATIC=0.55 \
bash "$SLIME_DIR/examples/ShopSimulator/run_rl.sh"
```

训练完成后，HF export 位于 `$RL_RUN_ROOT/hf/`，Megatron checkpoint 根目录为 `$RL_RUN_ROOT/checkpoints/`。同样以实际生成的 HF 子目录为准：

```bash
find "$RL_RUN_ROOT/hf" -mindepth 1 -maxdepth 1 -type d -print

export RL_HF_CHECKPOINT=/absolute/path/to/the/generated/rl/hf/export
export RL_MEGATRON_CHECKPOINT="$RL_RUN_ROOT/checkpoints"
```

### 实验 4：在 `official_test_200` 上做 k=1 评测

`shop_eval_official_k1.yaml` 对 200 个任务各 rollout 1 次。若要比较 Base、SFT 和 RL，按顺序运行下面三个命令；每次运行都会独立启动并清理 Ray，因此不要并行执行。

```bash
export BASE_EVAL_ROOT=/absolute/path/to/runs/eval_base_k1
EVAL_CHECKPOINT="$BASE_MEGATRON_CHECKPOINT" \
HF_CHECKPOINT="$BASE_HF_CHECKPOINT" \
RUN_ROOT="$BASE_EVAL_ROOT" \
SLIME_PYTHON="$SLIME_PYTHON" \
MEGATRON_DIR="$MEGATRON_DIR" \
PI_BIN="$PI_BIN" \
bash "$SLIME_DIR/examples/ShopSimulator/run_eval.sh"

export SFT_EVAL_ROOT=/absolute/path/to/runs/eval_sft_k1
EVAL_CHECKPOINT="$SFT_MEGATRON_CHECKPOINT" \
HF_CHECKPOINT="$SFT_HF_CHECKPOINT" \
RUN_ROOT="$SFT_EVAL_ROOT" \
SLIME_PYTHON="$SLIME_PYTHON" \
MEGATRON_DIR="$MEGATRON_DIR" \
PI_BIN="$PI_BIN" \
bash "$SLIME_DIR/examples/ShopSimulator/run_eval.sh"

export RL_EVAL_ROOT=/absolute/path/to/runs/eval_rl_k1
EVAL_CHECKPOINT="$RL_MEGATRON_CHECKPOINT" \
HF_CHECKPOINT="$RL_HF_CHECKPOINT" \
RUN_ROOT="$RL_EVAL_ROOT" \
SLIME_PYTHON="$SLIME_PYTHON" \
MEGATRON_DIR="$MEGATRON_DIR" \
PI_BIN="$PI_BIN" \
bash "$SLIME_DIR/examples/ShopSimulator/run_eval.sh"
```

每次评测的完整结果分别写入对应 `RUN_ROOT/eval_results.json`，其中包含 `r_loose`、`r_hard`、严格成功率、正奖励 pass@1、终止原因和逐任务记录。这里只运行评测，不会更新模型权重。

## 当前结构

- `slime/`：相对上述固定官方基线修改后的 Slime 源码快照；ShopSimulator 示例仅保留四条主流程所需代码、配置和数据。
- `shopsimulator_patch/shopsimulator-slime-integration.patch`：基于固定上游 revision 的单一 ShopSimulator 补丁；不分发上游完整源码。
- `README.md`：项目说明、相对基线的修改以及四个实验的完整运行流程。
- `CHANGES.md`：相对上游修改的摘要。
- `THIRD_PARTY.md`：第三方项目的来源、固定版本和许可边界。
- `repro.lock.json`：本项目使用的主要依赖与 revision 记录。
- `LICENSE`：本项目原创内容采用的 MIT License。

## 重要说明

ShopSimulator 上游当前未声明明确的软件再分发许可证，因此本临时结构不包含其完整源码。来源、固定 revision 和许可证状态见 `THIRD_PARTY.md`。

根目录 MIT License 仅覆盖本项目有权许可的原创代码和文档，不会改变第三方组件的许可证。内置 Slime 快照继续遵循 `slime/LICENSE` 中的 Apache-2.0；ShopSimulator 补丁也不授予对其上游源码的额外权利。

模型权重、大型训练数据、原始日志和 rollout dump 不进入 GitHub 仓库。两套训练后的 HF 模型已通过本文开头列出的公开 Hugging Face 仓库提供；历史 SFT 数据集为私有归档，原始日志和 rollout dump 不对外发布。
