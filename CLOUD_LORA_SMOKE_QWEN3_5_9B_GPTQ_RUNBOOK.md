# 云端操作手册：wizardeur/Qwen3.5-9B-GPTQ-marlin 手动 LoRA 冒烟测试

这份手册的目标是：在一台云端 Linux + NVIDIA GPU 服务器上，先 `git clone` 整个项目，然后按顺序执行命令，完成一次 Qwen3.5-9B-GPTQ LoRA 冒烟训练。

默认目标模型：

- `wizardeur/Qwen3.5-9B-GPTQ-marlin`

冒烟通过标准：

- 数据能下载、预处理、注册到 ms-swift。
- LoRA 训练能正常启动并持续输出 loss。
- `outputs/checkpoints/...` 下至少出现一个 checkpoint 或 adapter 产物。

## 0) 服务器基础检查

在云端服务器执行：

```bash
nvidia-smi
python3 --version
git --version
```

建议环境：

- Linux + NVIDIA GPU。
- Python >= 3.10。
- 已安装 `git`、`python3-venv`。

如果缺少基础工具（Ubuntu/Debian）：

```bash
sudo apt-get update
sudo apt-get install -y git python3-venv
```

## 1) 拉取或更新项目代码

第一次部署：

```bash
git clone https://github.com/zzjw99/-QA-.git
cd -- "-QA-"
```

如果云端已经有这个仓库：

```bash
cd /path/to/-QA-
git pull --ff-only
```

后续所有命令都默认在项目根目录执行。

## 2) 创建 Python 虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install -U pip setuptools wheel
```

## 3) 安装 CUDA 版 PyTorch

先确认你的服务器 CUDA 版本：

```bash
nvidia-smi
```

如果环境里还没有可用的 CUDA 版 PyTorch，请按 CUDA 版本安装。示例（CUDA 12.4）：

```bash
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
```

安装后检查：

```bash
python - <<'PY'
import torch
print("torch_version=", torch.__version__)
print("cuda_available=", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu=", torch.cuda.get_device_name(0))
PY
```

如果 `cuda_available=False`，先不要继续训练，优先修 PyTorch/CUDA 环境。

## 4) 安装项目与训练依赖

```bash
pip install -r requirements.txt

pip install -U ms-swift
pip install -U "transformers>=4.57.0" "accelerate>=1.2.0" "peft>=0.13.2" "datasets>=3.0.0" "huggingface_hub>=0.30.0"
pip install -U "qwen-vl-utils[decord]"

# GPTQ 运行时，至少装其中一个；两个都装也可以。
pip install -U gptqmodel || true
pip install -U auto-gptq || true
```

检查 `swift` 命令：

```bash
swift --help | head
```

## 5) 设置 Hugging Face 镜像与缓存

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=${HF_HOME:-$PWD/.hf_home}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$PWD/.hf_home/hub}
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE"
```

如果你有 Hugging Face Token，可选登录：

```bash
# export HF_TOKEN=<your_token>
# huggingface-cli login --token "$HF_TOKEN"
```

## 6) 下载 GPTQ 模型到本地磁盘

```bash
export MODEL_DIR=${MODEL_DIR:-$PWD/models/Qwen3.5-9B-GPTQ-marlin}
mkdir -p "$MODEL_DIR"

huggingface-cli download wizardeur/Qwen3.5-9B-GPTQ-marlin \
  --repo-type model \
  --local-dir "$MODEL_DIR" \
  --local-dir-use-symlinks False
```

如果 `huggingface-cli download` 不可用，可用 Python 方式：

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="wizardeur/Qwen3.5-9B-GPTQ-marlin",
    repo_type="model",
    local_dir="models/Qwen3.5-9B-GPTQ-marlin",
    local_dir_use_symlinks=False,
    endpoint="https://hf-mirror.com",
)
print("Model download complete")
PY

export MODEL_DIR=$PWD/models/Qwen3.5-9B-GPTQ-marlin
```

检查模型配置：

```bash
python - <<'PY'
import json
import os
from pathlib import Path

model_dir = Path(os.environ["MODEL_DIR"])
cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
print("model_dir:", model_dir)
print("architectures:", cfg.get("architectures"))
print("model_type:", cfg.get("model_type"))
print("has_vision_config:", "vision_config" in cfg)
print("quant_method:", (cfg.get("quantization_config") or {}).get("quant_method"))
PY
```

## 7) 下载 UrbanVideo-Bench 冒烟数据

这里采样 200 条记录，并下载对应视频，适合先验证训练链路。

```bash
python scripts/download_urbanvideo_bench.py \
  --dataset-id EmbodiedCity/UrbanVideo-Bench \
  --local-dir data/raw/urbanvideo_bench \
  --hf-endpoint https://hf-mirror.com \
  --sample-records 200 \
  --sample-strategy stratified \
  --sample-seed 42
```

检查视频是否下载到本地：

```bash
python - <<'PY'
from pathlib import Path

video_dir = Path("data/raw/urbanvideo_bench/videos")
count = sum(1 for p in video_dir.rglob("*") if p.is_file()) if video_dir.exists() else 0
print("downloaded_video_files=", count)
PY
```

## 8) 预处理数据并注册到 ms-swift

```bash
python scripts/prepare_urbanvideo_for_qwenvl.py \
  --raw-root data/raw/urbanvideo_bench \
  --output-root data/processed/urbanvideo_bench \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1
```

```bash
python scripts/register_dataset_in_ms_swift.py \
  --data-root data/raw/urbanvideo_bench \
  --train-jsonl data/processed/urbanvideo_bench/train.jsonl \
  --val-jsonl data/processed/urbanvideo_bench/val.jsonl \
  --test-jsonl data/processed/urbanvideo_bench/test.jsonl \
  --output-dir data/processed/urbanvideo_bench/ms_swift \
  --dataset-info-path outputs/ms_swift/dataset_info.json \
  --dataset-prefix urbanvideo
```

检查注册结果：

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("outputs/ms_swift/dataset_info.json")
info = json.loads(path.read_text(encoding="utf-8"))
print(json.dumps(info, ensure_ascii=False, indent=2))
PY
```

## 9) 先 dry-run 打印训练命令

这一步只打印 ms-swift 训练命令，不真正启动训练，方便检查路径和参数。

```bash
DRY_RUN=1 \
MODEL_NAME_OR_PATH="$MODEL_DIR" \
DATASET_INFO_PATH=outputs/ms_swift/dataset_info.json \
TRAIN_DATASET=urbanvideo_train \
VAL_DATASET=urbanvideo_val \
NUM_TRAIN_EPOCHS=1 \
MAX_STEPS=30 \
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
PER_DEVICE_EVAL_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=2 \
MAX_LENGTH=2048 \
DATALOADER_NUM_WORKERS=2 \
DATASET_NUM_PROC=2 \
SAVE_STEPS=20 \
EVAL_STEPS=20 \
LOGGING_STEPS=5 \
OUTPUT_DIR=outputs/checkpoints/ms_swift_smoke_qwen3_5_9b_gptq \
bash scripts/train_ms_swift_lora.sh
```

## 10) 启动 GPTQ LoRA 冒烟训练

```bash
mkdir -p outputs/logs
```

```bash
MODEL_NAME_OR_PATH="$MODEL_DIR" \
DATASET_INFO_PATH=outputs/ms_swift/dataset_info.json \
TRAIN_DATASET=urbanvideo_train \
VAL_DATASET=urbanvideo_val \
NUM_TRAIN_EPOCHS=1 \
MAX_STEPS=30 \
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
PER_DEVICE_EVAL_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=2 \
MAX_LENGTH=2048 \
DATALOADER_NUM_WORKERS=2 \
DATASET_NUM_PROC=2 \
SAVE_STEPS=20 \
EVAL_STEPS=20 \
LOGGING_STEPS=5 \
OUTPUT_DIR=outputs/checkpoints/ms_swift_smoke_qwen3_5_9b_gptq \
bash scripts/train_ms_swift_lora.sh 2>&1 | tee outputs/logs/ms_swift_smoke_qwen3_5_9b_gptq.log
```

如果训练能持续输出 loss，并最终保存 adapter/checkpoint，就说明云端 LoRA 训练链路已跑通。

## 11) 检查冒烟训练产物

```bash
find outputs/checkpoints/ms_swift_smoke_qwen3_5_9b_gptq -maxdepth 4 \
  \( -type d -name "checkpoint-*" -o -type f -name "adapter_config.json" -o -type f -name "adapter_model.safetensors" \) \
  -print
```

查看最近日志：

```bash
tail -n 100 outputs/logs/ms_swift_smoke_qwen3_5_9b_gptq.log
```

通过标准：

- 日志里出现训练 step 和 loss。
- 上面的 `find` 命令能找到 checkpoint 或 adapter 文件。

## 12) 如果 GPTQ 训练失败：切换非 GPTQ 基座

GPTQ 模型做 LoRA 训练可能会因为量化后端、CUDA、ms-swift 或 transformers 版本不匹配而失败。如果错误明显来自 GPTQ/quantization/backend，直接用非 GPTQ VL 基座验证项目训练链路：

```bash
MODEL_NAME_OR_PATH=Qwen/Qwen3.5-VL-7B-Instruct \
DATASET_INFO_PATH=outputs/ms_swift/dataset_info.json \
TRAIN_DATASET=urbanvideo_train \
VAL_DATASET=urbanvideo_val \
NUM_TRAIN_EPOCHS=1 \
MAX_STEPS=30 \
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
PER_DEVICE_EVAL_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=2 \
MAX_LENGTH=2048 \
DATALOADER_NUM_WORKERS=2 \
DATASET_NUM_PROC=2 \
SAVE_STEPS=20 \
EVAL_STEPS=20 \
LOGGING_STEPS=5 \
OUTPUT_DIR=outputs/checkpoints/ms_swift_smoke_qwen3_5_vl_7b \
bash scripts/train_ms_swift_lora.sh 2>&1 | tee outputs/logs/ms_swift_smoke_qwen3_5_vl_7b.log
```

检查回退模型产物：

```bash
find outputs/checkpoints/ms_swift_smoke_qwen3_5_vl_7b -maxdepth 4 \
  \( -type d -name "checkpoint-*" -o -type f -name "adapter_config.json" -o -type f -name "adapter_model.safetensors" \) \
  -print
```

## 13) 可选：使用一键冒烟脚本

如果你想少复制命令，也可以用项目里的自动化脚本完成同样流程。

先 dry-run：

```bash
bash scripts/cloud_lora_smoke_test.sh --dry-run
```

执行完整冒烟：

```bash
bash scripts/cloud_lora_smoke_test.sh \
  --install-deps \
  --install-gptq-deps \
  --download-model \
  --sample-records 200 \
  --max-steps 30
```

脚本输出：

- 日志：`outputs/logs/cloud_lora_smoke_*.log`
- 汇总：`outputs/reports/cloud_lora_smoke_summary_*.json`
- 训练产物：`outputs/checkpoints/ms_swift_smoke_qwen3_5_lora`

## 14) 可选：后续全量下载数据集

冒烟通过后，如果要做正式训练，再全量下载：

```bash
python scripts/download_urbanvideo_bench.py \
  --dataset-id EmbodiedCity/UrbanVideo-Bench \
  --local-dir data/raw/urbanvideo_bench \
  --hf-endpoint https://hf-mirror.com \
  --full-download
```
