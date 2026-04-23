# 云端操作手册：wizardeur/Qwen3.5-9B-GPTQ-marlin 手动 LoRA 冒烟测试

本文件用于云端服务器手动执行命令。
不依赖 Codex agent。

## 0) 适用范围与兼容性

- 目标模型：wizardeur/Qwen3.5-9B-GPTQ-marlin
- 该模型配置包含 vision/video token 与 vision_config，可用于本项目多模态链路。
- 该模型为 GPTQ 量化版本，可尝试 LoRA 训练，但依赖后端与运行时版本匹配。
- 如果出现 GPTQ 后端错误，请使用第 9 节回退方案。

## 1) 进入项目并创建环境

在云端 Linux 服务器执行：

python3 -m venv .venv
source .venv/bin/activate

python -m pip install -U pip setuptools wheel
pip install -r requirements.txt

## 2) 安装训练/运行依赖

pip install -U ms-swift
pip install -U "transformers>=4.57.0" "accelerate>=1.2.0" "peft>=0.13.2" "datasets>=3.0.0" "huggingface_hub>=0.30.0"
pip install -U qwen-vl-utils[decord]

# GPTQ 运行时（至少安装其中一个，两个都装也可以）
pip install -U gptqmodel || true
pip install -U auto-gptq || true

## 3) 设置镜像与缓存（推荐）

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=${HF_HOME:-$PWD/.hf_home}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$PWD/.hf_home/hub}
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE"

## 4) 下载模型到本地磁盘

MODEL_DIR=${MODEL_DIR:-$PWD/models/Qwen3.5-9B-GPTQ-marlin}
mkdir -p "$MODEL_DIR"

# 方式 A：huggingface-cli
huggingface-cli download wizardeur/Qwen3.5-9B-GPTQ-marlin \
  --repo-type model \
  --local-dir "$MODEL_DIR" \
  --local-dir-use-symlinks False

# 方式 B：Python snapshot_download（CLI 不可用时）
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

## 5) 快速检查模型配置

python - <<'PY'
import json
from pathlib import Path
cfg = json.loads(Path("models/Qwen3.5-9B-GPTQ-marlin/config.json").read_text(encoding="utf-8"))
print("architectures:", cfg.get("architectures"))
print("model_type:", cfg.get("model_type"))
print("has_vision_config:", "vision_config" in cfg)
print("quant_method:", (cfg.get("quantization_config") or {}).get("quant_method"))
PY

## 6) 下载 UrbanVideo-Bench（冒烟模式）

python scripts/download_urbanvideo_bench.py \
  --dataset-id EmbodiedCity/UrbanVideo-Bench \
  --local-dir data/raw/urbanvideo_bench \
  --hf-endpoint https://hf-mirror.com \
  --sample-records 200 \
  --sample-strategy stratified \
  --sample-seed 42

## 7) 预处理数据并注册到 ms-swift

python scripts/prepare_urbanvideo_for_qwenvl.py \
  --raw-root data/raw/urbanvideo_bench \
  --output-root data/processed/urbanvideo_bench \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1

python scripts/register_dataset_in_ms_swift.py \
  --data-root data/raw/urbanvideo_bench \
  --train-jsonl data/processed/urbanvideo_bench/train.jsonl \
  --val-jsonl data/processed/urbanvideo_bench/val.jsonl \
  --test-jsonl data/processed/urbanvideo_bench/test.jsonl \
  --output-dir data/processed/urbanvideo_bench/ms_swift \
  --dataset-info-path outputs/ms_swift/dataset_info.json \
  --dataset-prefix urbanvideo

## 8) 使用本地模型路径执行 LoRA 冒烟训练

chmod +x scripts/train_ms_swift_lora.sh

MODEL_NAME_OR_PATH="$MODEL_DIR" \
DATASET_INFO_PATH=outputs/ms_swift/dataset_info.json \
TRAIN_DATASET=urbanvideo_train \
VAL_DATASET=urbanvideo_val \
NUM_TRAIN_EPOCHS=1 \
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
bash scripts/train_ms_swift_lora.sh --max_steps 30

## 9) GPTQ 训练失败时的回退方案

如果训练报量化后端错误，请切换到非 GPTQ 的 VL 基座模型：

MODEL_NAME_OR_PATH=Qwen/Qwen3.5-VL-7B-Instruct \
DATASET_INFO_PATH=outputs/ms_swift/dataset_info.json \
TRAIN_DATASET=urbanvideo_train \
VAL_DATASET=urbanvideo_val \
NUM_TRAIN_EPOCHS=1 \
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
bash scripts/train_ms_swift_lora.sh --max_steps 30

## 10) 冒烟通过标准

- 训练能够正常启动并持续输出 loss。
- 至少生成一个 checkpoint 目录。

find outputs/checkpoints -maxdepth 3 -type d -name "checkpoint-*"

## 11) 可选：后续全量下载数据集

python scripts/download_urbanvideo_bench.py \
  --dataset-id EmbodiedCity/UrbanVideo-Bench \
  --local-dir data/raw/urbanvideo_bench \
  --hf-endpoint https://hf-mirror.com \
  --full-download
