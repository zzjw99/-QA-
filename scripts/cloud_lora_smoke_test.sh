#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Cloud LoRA smoke test for UrbanVideo-Bench + ms-swift.

Usage:
  bash scripts/cloud_lora_smoke_test.sh [options] [-- extra swift args]

Common options:
  --dry-run              Print commands and write a summary without executing them.
  --install-deps         Install Python runtime/training dependencies except torch.
  --install-gptq-deps    Install GPTQ runtime packages. Best used with GPTQ models.
  --download-model       Download MODEL_REPO to MODEL_DIR before training.
  --skip-data-download   Reuse existing raw data.
  --skip-prepare         Reuse existing processed JSONL and dataset_info.json.
  --skip-train           Stop after data/model/runtime validation.
  --full-download        Download the full dataset instead of sample records.
  --model VALUE          Override MODEL_NAME_OR_PATH.
  --model-repo VALUE     Override MODEL_REPO used by --download-model.
  --model-dir DIR        Override MODEL_DIR used by --download-model.
  --sample-records N     Number of records for smoke data download.
  --max-videos N         Optional cap on sampled videos.
  --max-steps N          Training steps for the smoke run.
  --output-dir DIR       Training output directory.
  -h, --help             Show this help.

Environment overrides are also supported. Useful variables:
  MODEL_NAME_OR_PATH, DATASET_ID, HF_ENDPOINT, SAMPLE_RECORDS, MAX_STEPS,
  OUTPUT_DIR, DATA_ROOT, PROCESSED_ROOT, DATASET_INFO_PATH, TRAIN_DATASET,
  VAL_DATASET, ALLOW_NO_CUDA.
EOF
}

is_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

section() {
  echo
  echo "========== $* =========="
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${ROOT_DIR}"

STAMP="$(date +%Y%m%d_%H%M%S)"

PYTHON="${PYTHON:-python3}"
if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  PYTHON="python"
fi

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_HOME="${HF_HOME:-${ROOT_DIR}/.hf_home}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"

DATASET_ID="${DATASET_ID:-EmbodiedCity/UrbanVideo-Bench}"
DATA_ROOT="${DATA_ROOT:-data/raw/urbanvideo_bench}"
PROCESSED_ROOT="${PROCESSED_ROOT:-data/processed/urbanvideo_bench}"
DATASET_INFO_PATH="${DATASET_INFO_PATH:-outputs/ms_swift/dataset_info.json}"
DATASET_PREFIX="${DATASET_PREFIX:-urbanvideo}"
TRAIN_DATASET="${TRAIN_DATASET:-${DATASET_PREFIX}_train}"
VAL_DATASET="${VAL_DATASET:-${DATASET_PREFIX}_val}"
MS_SWIFT_OUTPUT_DIR="${MS_SWIFT_OUTPUT_DIR:-${PROCESSED_ROOT}/ms_swift}"

MODEL_REPO="${MODEL_REPO:-wizardeur/Qwen3.5-9B-GPTQ-marlin}"
MODEL_DIR="${MODEL_DIR:-models/Qwen3.5-9B-GPTQ-marlin}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${MODEL_REPO}}"

SAMPLE_RECORDS="${SAMPLE_RECORDS:-200}"
SAMPLE_STRATEGY="${SAMPLE_STRATEGY:-stratified}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
MAX_VIDEOS="${MAX_VIDEOS:-}"
FULL_DOWNLOAD="${FULL_DOWNLOAD:-0}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/checkpoints/ms_swift_smoke_qwen3_5_lora}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
REPORT_DIR="${REPORT_DIR:-outputs/reports}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/cloud_lora_smoke_${STAMP}.log}"
SUMMARY_PATH="${SUMMARY_PATH:-${REPORT_DIR}/cloud_lora_smoke_summary_${STAMP}.json}"

INSTALL_DEPS="${INSTALL_DEPS:-0}"
INSTALL_GPTQ_DEPS="${INSTALL_GPTQ_DEPS:-0}"
DOWNLOAD_MODEL="${DOWNLOAD_MODEL:-0}"
SKIP_DATA_DOWNLOAD="${SKIP_DATA_DOWNLOAD:-0}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_NO_CUDA="${ALLOW_NO_CUDA:-0}"

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-30}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-2}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
SAVE_STEPS="${SAVE_STEPS:-20}"
EVAL_STEPS="${EVAL_STEPS:-20}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
DEEPSPEED="${DEEPSPEED:-}"
USE_HF="${USE_HF:-true}"

TRAIN_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --install-deps)
      INSTALL_DEPS=1
      shift
      ;;
    --install-gptq-deps)
      INSTALL_GPTQ_DEPS=1
      shift
      ;;
    --download-model)
      DOWNLOAD_MODEL=1
      shift
      ;;
    --skip-data-download)
      SKIP_DATA_DOWNLOAD=1
      shift
      ;;
    --skip-prepare)
      SKIP_PREPARE=1
      shift
      ;;
    --skip-train)
      SKIP_TRAIN=1
      shift
      ;;
    --full-download)
      FULL_DOWNLOAD=1
      shift
      ;;
    --model)
      [[ $# -ge 2 ]] || die "--model requires a value"
      MODEL_NAME_OR_PATH="$2"
      shift 2
      ;;
    --model-repo)
      [[ $# -ge 2 ]] || die "--model-repo requires a value"
      MODEL_REPO="$2"
      shift 2
      ;;
    --model-dir)
      [[ $# -ge 2 ]] || die "--model-dir requires a value"
      MODEL_DIR="$2"
      shift 2
      ;;
    --sample-records)
      [[ $# -ge 2 ]] || die "--sample-records requires a value"
      SAMPLE_RECORDS="$2"
      shift 2
      ;;
    --max-videos)
      [[ $# -ge 2 ]] || die "--max-videos requires a value"
      MAX_VIDEOS="$2"
      shift 2
      ;;
    --max-steps)
      [[ $# -ge 2 ]] || die "--max-steps requires a value"
      MAX_STEPS="$2"
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || die "--output-dir requires a value"
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      TRAIN_ARGS+=("$@")
      break
      ;;
    *)
      die "Unknown option: $1"
      ;;
  esac
done

if [[ -n "${TRAIN_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  TRAIN_ARGS+=(${TRAIN_EXTRA_ARGS})
fi

mkdir -p "${LOG_DIR}" "${REPORT_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

START_EPOCH="$(date +%s)"
export ROOT_DIR HF_ENDPOINT HF_HOME HUGGINGFACE_HUB_CACHE DATASET_ID DATA_ROOT
export PROCESSED_ROOT DATASET_INFO_PATH DATASET_PREFIX TRAIN_DATASET VAL_DATASET
export MODEL_REPO MODEL_DIR MODEL_NAME_OR_PATH OUTPUT_DIR LOG_FILE SUMMARY_PATH
export SAMPLE_RECORDS SAMPLE_STRATEGY SAMPLE_SEED MAX_VIDEOS FULL_DOWNLOAD
export MAX_STEPS DRY_RUN SKIP_TRAIN ALLOW_NO_CUDA START_EPOCH

write_summary() {
  local status="$1"
  local exit_code="$2"
  if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    return 0
  fi

  SUMMARY_STATUS="${status}" SUMMARY_EXIT_CODE="${exit_code}" "${PYTHON}" - <<'PY' || true
import json
import os
import time
from pathlib import Path

def env(name, default=""):
    return os.environ.get(name, default)

output_dir = Path(env("OUTPUT_DIR"))
checkpoints = []
if output_dir.exists():
    for path in output_dir.rglob("*"):
        if path.is_dir() and path.name.startswith("checkpoint-"):
            checkpoints.append(path.as_posix())
        elif path.is_file() and path.name in {"adapter_config.json", "adapter_model.safetensors"}:
            checkpoints.append(path.as_posix())

start = int(env("START_EPOCH", str(int(time.time()))))
summary = {
    "status": env("SUMMARY_STATUS"),
    "exit_code": int(env("SUMMARY_EXIT_CODE", "0")),
    "started_at_epoch": start,
    "finished_at_epoch": int(time.time()),
    "duration_seconds": int(time.time()) - start,
    "root_dir": env("ROOT_DIR"),
    "log_file": env("LOG_FILE"),
    "dataset": {
        "dataset_id": env("DATASET_ID"),
        "data_root": env("DATA_ROOT"),
        "processed_root": env("PROCESSED_ROOT"),
        "dataset_info_path": env("DATASET_INFO_PATH"),
        "train_dataset": env("TRAIN_DATASET"),
        "val_dataset": env("VAL_DATASET"),
        "sample_records": env("SAMPLE_RECORDS"),
        "sample_strategy": env("SAMPLE_STRATEGY"),
        "sample_seed": env("SAMPLE_SEED"),
        "max_videos": env("MAX_VIDEOS"),
        "full_download": env("FULL_DOWNLOAD"),
    },
    "model": {
        "model_name_or_path": env("MODEL_NAME_OR_PATH"),
        "model_repo": env("MODEL_REPO"),
        "model_dir": env("MODEL_DIR"),
    },
    "training": {
        "output_dir": env("OUTPUT_DIR"),
        "max_steps": env("MAX_STEPS"),
        "skip_train": env("SKIP_TRAIN"),
        "dry_run": env("DRY_RUN"),
        "artifacts": sorted(set(checkpoints)),
    },
}

summary_path = Path(env("SUMMARY_PATH"))
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[INFO] Summary written to: {summary_path}")
PY
}

on_error() {
  local code="$?"
  trap - ERR
  echo
  echo "[ERROR] Cloud LoRA smoke test failed with exit code ${code}."
  if [[ "${MODEL_NAME_OR_PATH,,}" == *gptq* ]]; then
    echo "[HINT] GPTQ LoRA can fail because of quantization backend compatibility."
    echo "[HINT] Retry with: --model Qwen/Qwen3.5-VL-7B-Instruct"
  fi
  write_summary "failed" "${code}"
  exit "${code}"
}
trap on_error ERR

run() {
  printf '[CMD]'
  printf ' %q' "$@"
  echo
  if is_true "${DRY_RUN}"; then
    return 0
  fi
  "$@"
}

print_config() {
  section "Configuration"
  cat <<EOF
root_dir=${ROOT_DIR}
python=${PYTHON}
hf_endpoint=${HF_ENDPOINT}
dataset_id=${DATASET_ID}
sample_records=${SAMPLE_RECORDS}
data_root=${DATA_ROOT}
processed_root=${PROCESSED_ROOT}
dataset_info_path=${DATASET_INFO_PATH}
model_name_or_path=${MODEL_NAME_OR_PATH}
model_repo=${MODEL_REPO}
model_dir=${MODEL_DIR}
output_dir=${OUTPUT_DIR}
max_steps=${MAX_STEPS}
dry_run=${DRY_RUN}
skip_train=${SKIP_TRAIN}
log_file=${LOG_FILE}
summary_path=${SUMMARY_PATH}
EOF
}

base_preflight() {
  section "Base preflight"
  command -v "${PYTHON}" >/dev/null 2>&1 || die "Python not found. Set PYTHON=/path/to/python."
  "${PYTHON}" --version

  if command -v git >/dev/null 2>&1; then
    git --version
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
  else
    echo "[WARN] nvidia-smi not found."
    if ! is_true "${DRY_RUN}" && ! is_true "${SKIP_TRAIN}" && ! is_true "${ALLOW_NO_CUDA}"; then
      die "GPU check failed. Set ALLOW_NO_CUDA=1 only for non-training validation."
    fi
  fi

  mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${OUTPUT_DIR}"
  export HF_ENDPOINT HF_HOME HUGGINGFACE_HUB_CACHE
}

install_deps() {
  section "Dependency setup"
  if is_true "${INSTALL_DEPS}"; then
    run "${PYTHON}" -m pip install -r requirements.txt
    run "${PYTHON}" -m pip install -U ms-swift "transformers>=4.57.0" "accelerate>=1.2.0" "peft>=0.13.2" "datasets>=3.0.0" "huggingface_hub>=0.30.0" "qwen-vl-utils[decord]"
  else
    echo "[INFO] INSTALL_DEPS=0, skipping pip installs."
  fi

  if is_true "${INSTALL_GPTQ_DEPS}"; then
    if ! run "${PYTHON}" -m pip install -U gptqmodel auto-gptq; then
      echo "[WARN] GPTQ dependency install failed. Continuing so the main error stays visible."
    fi
  fi
}

training_runtime_preflight() {
  section "Training runtime preflight"
  if is_true "${SKIP_TRAIN}" || is_true "${DRY_RUN}"; then
    echo "[INFO] Training runtime checks are relaxed because SKIP_TRAIN=${SKIP_TRAIN}, DRY_RUN=${DRY_RUN}."
    return 0
  fi

  command -v swift >/dev/null 2>&1 || die "'swift' command not found. Run with --install-deps or install ms-swift."

  "${PYTHON}" - <<'PY'
import os
import sys

try:
    import torch
except Exception as exc:
    print(f"[ERROR] torch import failed: {exc}")
    print("[HINT] Install a CUDA PyTorch build that matches your server before training.")
    sys.exit(1)

print("torch_version=", torch.__version__)
print("cuda_available=", torch.cuda.is_available())
if not torch.cuda.is_available():
    allow_no_cuda = os.environ.get("ALLOW_NO_CUDA", "0").lower() in {"1", "true", "yes", "on"}
    if allow_no_cuda:
        print("[WARN] CUDA is unavailable, but ALLOW_NO_CUDA=1 so runtime preflight continues.")
        sys.exit(0)
    sys.exit(2)
print("gpu_count=", torch.cuda.device_count())
print("gpu0=", torch.cuda.get_device_name(0))
PY
}

download_model_if_requested() {
  section "Model check"
  if is_true "${DOWNLOAD_MODEL}"; then
    run "${PYTHON}" - <<PY
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="${MODEL_REPO}",
    repo_type="model",
    local_dir="${MODEL_DIR}",
    local_dir_use_symlinks=False,
    endpoint="${HF_ENDPOINT}",
)
print("Model downloaded to: ${MODEL_DIR}")
PY
    if [[ "${MODEL_NAME_OR_PATH}" == "${MODEL_REPO}" ]]; then
      MODEL_NAME_OR_PATH="${MODEL_DIR}"
      export MODEL_NAME_OR_PATH
    fi
  else
    echo "[INFO] DOWNLOAD_MODEL=0, using model value directly."
  fi

  if [[ -f "${MODEL_NAME_OR_PATH}/config.json" ]]; then
    "${PYTHON}" - "${MODEL_NAME_OR_PATH}/config.json" <<'PY'
import json
import sys
from pathlib import Path

cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print("architectures=", cfg.get("architectures"))
print("model_type=", cfg.get("model_type"))
print("has_vision_config=", "vision_config" in cfg)
print("quant_method=", (cfg.get("quantization_config") or {}).get("quant_method"))
PY
  else
    echo "[INFO] No local config.json found for MODEL_NAME_OR_PATH; assuming a hub model id or ms-swift-resolvable path."
  fi
}

download_data() {
  section "Dataset download"
  if is_true "${SKIP_DATA_DOWNLOAD}"; then
    echo "[INFO] SKIP_DATA_DOWNLOAD=1, reusing existing raw data."
    return 0
  fi

  local args=(
    scripts/download_urbanvideo_bench.py
    --dataset-id "${DATASET_ID}"
    --local-dir "${DATA_ROOT}"
    --hf-endpoint "${HF_ENDPOINT}"
    --sample-strategy "${SAMPLE_STRATEGY}"
    --sample-seed "${SAMPLE_SEED}"
  )

  if is_true "${FULL_DOWNLOAD}"; then
    args+=(--full-download)
  else
    args+=(--sample-records "${SAMPLE_RECORDS}")
  fi

  if [[ -n "${MAX_VIDEOS}" ]]; then
    args+=(--max-videos "${MAX_VIDEOS}")
  fi

  run "${PYTHON}" "${args[@]}"
}

prepare_data() {
  section "Dataset preparation"
  if is_true "${SKIP_PREPARE}"; then
    echo "[INFO] SKIP_PREPARE=1, reusing processed JSONL and dataset_info.json."
    return 0
  fi

  run "${PYTHON}" scripts/prepare_urbanvideo_for_qwenvl.py \
    --raw-root "${DATA_ROOT}" \
    --output-root "${PROCESSED_ROOT}" \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1

  run "${PYTHON}" scripts/register_dataset_in_ms_swift.py \
    --data-root "${DATA_ROOT}" \
    --train-jsonl "${PROCESSED_ROOT}/train.jsonl" \
    --val-jsonl "${PROCESSED_ROOT}/val.jsonl" \
    --test-jsonl "${PROCESSED_ROOT}/test.jsonl" \
    --output-dir "${MS_SWIFT_OUTPUT_DIR}" \
    --dataset-info-path "${DATASET_INFO_PATH}" \
    --dataset-prefix "${DATASET_PREFIX}"
}

validate_dataset_artifacts() {
  section "Dataset artifact validation"
  if is_true "${DRY_RUN}"; then
    echo "[INFO] DRY_RUN=1, skipping artifact validation."
    return 0
  fi

  [[ -f "${PROCESSED_ROOT}/train.jsonl" ]] || die "Missing train JSONL: ${PROCESSED_ROOT}/train.jsonl"
  [[ -f "${PROCESSED_ROOT}/val.jsonl" ]] || die "Missing val JSONL: ${PROCESSED_ROOT}/val.jsonl"
  [[ -f "${DATASET_INFO_PATH}" ]] || die "Missing dataset_info.json: ${DATASET_INFO_PATH}"

  "${PYTHON}" - "${PROCESSED_ROOT}/train.jsonl" "${PROCESSED_ROOT}/val.jsonl" "${DATASET_INFO_PATH}" <<'PY'
import json
import sys
from pathlib import Path

for item in sys.argv[1:3]:
    path = Path(item)
    count = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    print(f"{path}: {count} records")
    if count <= 0:
        raise SystemExit(f"Empty JSONL: {path}")

info_path = Path(sys.argv[3])
info = json.loads(info_path.read_text(encoding="utf-8"))
names = [item.get("dataset_name") for item in info]
print(f"{info_path}: {names}")
PY
}

run_training() {
  section "LoRA smoke training"
  if is_true "${SKIP_TRAIN}"; then
    echo "[INFO] SKIP_TRAIN=1, not launching ms-swift training."
    return 0
  fi

  local train_env=(
    "MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
    "DATASET_INFO_PATH=${DATASET_INFO_PATH}"
    "TRAIN_DATASET=${TRAIN_DATASET}"
    "VAL_DATASET=${VAL_DATASET}"
    "OUTPUT_DIR=${OUTPUT_DIR}"
    "TORCH_DTYPE=${TORCH_DTYPE}"
    "NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}"
    "MAX_STEPS=${MAX_STEPS}"
    "PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}"
    "PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE}"
    "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"
    "MAX_LENGTH=${MAX_LENGTH}"
    "DATASET_NUM_PROC=${DATASET_NUM_PROC}"
    "DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS}"
    "SAVE_STEPS=${SAVE_STEPS}"
    "EVAL_STEPS=${EVAL_STEPS}"
    "LOGGING_STEPS=${LOGGING_STEPS}"
    "SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT}"
    "DEEPSPEED=${DEEPSPEED}"
    "USE_HF=${USE_HF}"
    "DRY_RUN=${DRY_RUN}"
  )

  run env "${train_env[@]}" bash scripts/train_ms_swift_lora.sh "${TRAIN_ARGS[@]}"
}

validate_training_artifacts() {
  section "Training artifact validation"
  if is_true "${SKIP_TRAIN}" || is_true "${DRY_RUN}"; then
    echo "[INFO] Training artifact validation skipped."
    return 0
  fi

  [[ -d "${OUTPUT_DIR}" ]] || die "Training output dir was not created: ${OUTPUT_DIR}"

  mapfile -t artifacts < <(
    find "${OUTPUT_DIR}" -maxdepth 4 \( -type d -name "checkpoint-*" -o -type f -name "adapter_config.json" -o -type f -name "adapter_model.safetensors" \) | sort
  )

  if [[ "${#artifacts[@]}" -eq 0 ]]; then
    die "No checkpoint or adapter artifact found under ${OUTPUT_DIR}. Increase MAX_STEPS or lower SAVE_STEPS."
  fi

  printf '[INFO] Found %s training artifacts:\n' "${#artifacts[@]}"
  printf '  %s\n' "${artifacts[@]}"
}

print_config
base_preflight
install_deps
training_runtime_preflight
download_model_if_requested
download_data
prepare_data
validate_dataset_artifacts
run_training
validate_training_artifacts

if is_true "${DRY_RUN}"; then
  write_summary "dry_run" 0
  echo
  echo "[OK] Dry run complete. Log: ${LOG_FILE}"
else
  write_summary "passed" 0
  echo
  echo "[OK] Cloud LoRA smoke test passed. Log: ${LOG_FILE}"
fi
