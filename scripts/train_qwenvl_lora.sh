#!/usr/bin/env bash
set -euo pipefail

# Example launch script for low-budget Qwen2.5-VL-7B LoRA training.
# Run this on your Linux cloud machine after dataset registration.
# 中文说明：
# 1) 这是云端训练入口脚本，默认按低预算配置跑 7B LoRA。
# 2) 你可以用环境变量覆盖下面的默认值，不需要改脚本本体。
# 3) 使用前需要先完成：数据转换 + 数据集注册。

# Qwen 官方仓库目录（包含 qwen-vl-finetune）
QWEN_REPO_DIR=${QWEN_REPO_DIR:-third_party/Qwen2.5-VL}
FRAMEWORK_DIR="${QWEN_REPO_DIR}/qwen-vl-finetune"

# 基础模型与数据别名
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}
DATASET_USE=${DATASET_USE:-urbanvideo_train%100}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/checkpoints/qwen2_5_vl_7b_lora}
CACHE_DIR=${CACHE_DIR:-outputs/cache}

# 单机多卡通信参数
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}

mkdir -p "${OUTPUT_DIR}" "${CACHE_DIR}"

cd "${FRAMEWORK_DIR}"

# 训练主命令：参数保持与 Qwen 官方训练脚本风格一致
torchrun --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  qwenvl/train/train_qwen.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --tune_mm_llm True \
  --tune_mm_vision False \
  --tune_mm_mlp False \
  --dataset_use "${DATASET_USE}" \
  --output_dir "${OUTPUT_DIR}" \
  --cache_dir "${CACHE_DIR}" \
  --bf16 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-7 \
  --mm_projector_lr 1e-5 \
  --vision_tower_lr 1e-6 \
  --optim adamw_torch \
  --model_max_length 4096 \
  --data_flatten True \
  --video_fps 2 \
  --video_max_frames 8 \
  --video_min_frames 4 \
  --video_max_pixels 1664*28*28 \
  --video_min_pixels 256*28*28 \
  --num_train_epochs 3 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --weight_decay 0.01 \
  --logging_steps 10 \
  --save_steps 500 \
  --save_total_limit 3 \
  --lora_enable True \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.0

# 训练结束后，检查点默认在：${OUTPUT_DIR}
