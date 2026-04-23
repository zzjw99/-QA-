#!/usr/bin/env bash
set -euo pipefail

# Example launch script for ms-swift LoRA training on Qwen-VL.
# 中文说明：
# 1) 这是不改第三方源码的训练入口，依赖 custom dataset_info.json 进行数据注册。
# 2) 你可以通过环境变量覆盖默认参数。
# 3) 训练前请先执行 scripts/register_dataset_in_ms_swift.py。

MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}
DATASET_INFO_PATH=${DATASET_INFO_PATH:-outputs/ms_swift/dataset_info.json}
TRAIN_DATASET=${TRAIN_DATASET:-urbanvideo_train}
VAL_DATASET=${VAL_DATASET:-urbanvideo_val}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/checkpoints/ms_swift_qwen2_5_vl_7b_lora}

TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-3}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
LEARNING_RATE=${LEARNING_RATE:-2e-7}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
MAX_LENGTH=${MAX_LENGTH:-4096}
DATASET_NUM_PROC=${DATASET_NUM_PROC:-4}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}

TUNER_TYPE=${TUNER_TYPE:-lora}
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}
TARGET_MODULES=${TARGET_MODULES:-all-linear}

SAVE_STEPS=${SAVE_STEPS:-500}
EVAL_STEPS=${EVAL_STEPS:-500}
LOGGING_STEPS=${LOGGING_STEPS:-10}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-3}

DEEPSPEED=${DEEPSPEED:-}
USE_HF=${USE_HF:-true}

if ! command -v swift >/dev/null 2>&1; then
  echo "[ERROR] 'swift' command not found. Install with: pip install -U ms-swift"
  exit 1
fi

if [[ ! -f "${DATASET_INFO_PATH}" ]]; then
  echo "[ERROR] dataset_info file not found: ${DATASET_INFO_PATH}"
  echo "Run first: python scripts/register_dataset_in_ms_swift.py"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

cmd=(
  swift sft
  --model "${MODEL_NAME_OR_PATH}"
  --custom_dataset_info "${DATASET_INFO_PATH}"
  --dataset "${TRAIN_DATASET}"
  --tuner_type "${TUNER_TYPE}"
  --torch_dtype "${TORCH_DTYPE}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --learning_rate "${LEARNING_RATE}"
  --warmup_ratio "${WARMUP_RATIO}"
  --max_length "${MAX_LENGTH}"
  --dataset_num_proc "${DATASET_NUM_PROC}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --lora_rank "${LORA_RANK}"
  --lora_alpha "${LORA_ALPHA}"
  --lora_dropout "${LORA_DROPOUT}"
  --target_modules "${TARGET_MODULES}"
  --save_steps "${SAVE_STEPS}"
  --eval_steps "${EVAL_STEPS}"
  --logging_steps "${LOGGING_STEPS}"
  --save_total_limit "${SAVE_TOTAL_LIMIT}"
  --output_dir "${OUTPUT_DIR}"
  --use_hf "${USE_HF}"
)

if [[ -n "${VAL_DATASET}" ]]; then
  cmd+=(--val_dataset "${VAL_DATASET}")
fi

if [[ -n "${DEEPSPEED}" ]]; then
  cmd+=(--deepspeed "${DEEPSPEED}")
fi

if [[ "${TUNER_TYPE}" != "lora" ]]; then
  echo "[WARN] TUNER_TYPE=${TUNER_TYPE}, LoRA-specific args will still be passed."
fi

if [[ $# -gt 0 ]]; then
  cmd+=("$@")
fi

echo "[INFO] Running ms-swift SFT command:"
printf ' %q' "${cmd[@]}"
echo

"${cmd[@]}"

# 训练结束后，检查点默认在：${OUTPUT_DIR}
