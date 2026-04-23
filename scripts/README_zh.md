# scripts 目录中文说明

这个文档是给你“直接看脚本就能跑”的速查版，重点看每个脚本的输入、输出、常见命令。

云端推荐：优先使用 QwenVL 3.5（7B Instruct），备选 QwenVL 3.0（7B Instruct）。

## 1) download_urbanvideo_bench.py

作用：从 Hugging Face（默认使用 hf-mirror）下载 UrbanVideo-Bench 到本地。

常用命令：

```bash
python scripts/download_urbanvideo_bench.py \
  --dataset-id EmbodiedCity/UrbanVideo-Bench \
  --local-dir data/raw/urbanvideo_bench \
  --hf-endpoint https://hf-mirror.com \
  --sample-records 200
```

输出：

1. 数据文件（parquet/videos）。
2. `download_manifest.json`（下载清单）。

脚本默认会按 `question_category` 分层随机采样 200 条记录，再下载对应视频（用于快速验证 LoRA 链路）。

如果你只想按视频数做额外限制，可叠加：

```bash
python scripts/download_urbanvideo_bench.py \
  --dataset-id EmbodiedCity/UrbanVideo-Bench \
  --local-dir data/raw/urbanvideo_bench \
  --sample-records 200 \
  --max-videos 150 \
  --sample-seed 42
```

如果要全量下载（不采样）：

```bash
python scripts/download_urbanvideo_bench.py \
  --dataset-id EmbodiedCity/UrbanVideo-Bench \
  --local-dir data/raw/urbanvideo_bench \
  --full-download
```

采样模式额外输出：

1. `sampled_videos_manifest.json`（采样视频列表与缺失情况）。
2. `download_manifest.json` 中 `sample_mode` 字段（采样统计）。

## 2) prepare_urbanvideo_for_qwenvl.py

作用：把 parquet + 视频转成 QwenVL 的 JSONL，并按 `video_id` 切分 train/val/test。

常用命令：

```bash
python scripts/prepare_urbanvideo_for_qwenvl.py \
  --raw-root data/raw/urbanvideo_bench \
  --output-root data/processed/urbanvideo_bench \
  --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
```

输出：

1. `train.jsonl` / `val.jsonl` / `test.jsonl`。
2. `ground_truth_*.jsonl`。
3. `prepare_summary.json`。

## 3) setup_qwenvl_repo.ps1

作用：拉取或更新 QwenVL 官方仓库（推荐 3.0/3.5）。

常用命令：

```powershell
powershell ./scripts/setup_qwenvl_repo.ps1 -RepoDir third_party/Qwen3.5-VL -RepoUrl https://github.com/QwenLM/Qwen3.5-VL.git
powershell ./scripts/setup_qwenvl_repo.ps1 -RepoDir third_party/Qwen3.5-VL -RepoUrl https://github.com/QwenLM/Qwen3.5-VL.git -Pull
```

备选（3.0）：

```powershell
powershell ./scripts/setup_qwenvl_repo.ps1 -RepoDir third_party/Qwen3-VL -RepoUrl https://github.com/QwenLM/Qwen3-VL.git
```

## 4) register_dataset_in_ms_swift.py（推荐）

作用：把你处理后的 QwenVL JSONL 转成 ms-swift 格式，并生成 `dataset_info.json`。

常用命令：

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

默认数据集别名：`urbanvideo_train` / `urbanvideo_val` / `urbanvideo_test`。

## 5) train_ms_swift_lora.sh（推荐）

作用：使用 ms-swift 启动多模态 LoRA 训练，不需要修改第三方源码。

常用命令：

```bash
pip install -U ms-swift
bash scripts/train_ms_swift_lora.sh
```

带覆盖参数：

```bash
MODEL_NAME_OR_PATH=Qwen/Qwen3.5-VL-7B-Instruct \
DATASET_INFO_PATH=outputs/ms_swift/dataset_info.json \
TRAIN_DATASET=urbanvideo_train \
VAL_DATASET=urbanvideo_val \
DEEPSPEED=zero2 \
bash scripts/train_ms_swift_lora.sh
```

## 6) register_dataset_in_qwenvl.py（兼容旧链路）

作用：把你处理后的 JSONL 注册成 Qwen 框架可识别的数据集别名。

常用命令：

```bash
python scripts/register_dataset_in_qwenvl.py \
  --qwenvl-data-init third_party/Qwen3.5-VL/qwen-vl-finetune/qwenvl/data/__init__.py \
  --data-path data/raw/urbanvideo_bench \
  --dataset urbanvideo_train=data/processed/urbanvideo_bench/train.jsonl \
  --dataset urbanvideo_val=data/processed/urbanvideo_bench/val.jsonl
```

## 7) train_qwenvl_lora.sh（兼容旧链路）

作用：云端启动 LoRA 训练（默认低预算参数）。

常用命令：

```bash
bash scripts/train_qwenvl_lora.sh
```

带覆盖参数：

```bash
MODEL_NAME_OR_PATH=Qwen/Qwen3.5-VL-7B-Instruct \
DATASET_USE=urbanvideo_train%100 \
NPROC_PER_NODE=2 \
bash scripts/train_qwenvl_lora.sh
```

备选：`MODEL_NAME_OR_PATH=Qwen/Qwen3-VL-7B-Instruct`。

输出：`outputs/checkpoints/qwen_vl_7b_lora`（建议按 3.0/3.5 区分命名）。

## 8) infer_qwen2_5_vl_mcq.py

作用：在某个 split（通常 test）上跑推理，导出预测。

常用命令：

```bash
python scripts/infer_qwen2_5_vl_mcq.py \
  --model-name-or-path <模型路径> \
  --input-jsonl data/processed/urbanvideo_bench/test.jsonl \
  --data-root data/raw/urbanvideo_bench \
  --output-jsonl outputs/predictions/test_predictions.jsonl
```

高性能命令（Linux + GPU，建议比赛冲榜使用）：

```bash
pip install vllm outlines qwen-vl-utils[decord]

python scripts/infer_qwen2_5_vl_mcq.py \
  --backend vllm \
  --model-name-or-path <模型路径> \
  --input-jsonl data/processed/urbanvideo_bench/test.jsonl \
  --data-root data/raw/urbanvideo_bench \
  --output-jsonl outputs/predictions/test_predictions_vllm.jsonl \
  --batch-size 16 \
  --constrained-decoding
```

参数提示：

1. `--backend`：`transformers`（默认，兼容优先）或 `vllm`（吞吐优先）。
2. `--batch-size`：按显存调节，通常 8-32 有较好吞吐。
3. `--constrained-decoding`：将答案限制为题目允许字母，避免输出解释性长文本。

## 9) eval_mcq_accuracy.py

作用：计算 MCQ 准确率（总分 + 分类别）。

常用命令：

```bash
python scripts/eval_mcq_accuracy.py \
  --ground-truth data/processed/urbanvideo_bench/ground_truth_test.jsonl \
  --predictions outputs/predictions/test_predictions.jsonl \
  --report-path outputs/reports/test_eval_report.json
```

## 10) capability_gate.py

作用：根据阈值自动判定模型能力是否达标（go/no-go）。

常用命令：

```bash
python scripts/capability_gate.py \
  --report outputs/reports/test_eval_report.json \
  --output outputs/reports/test_gate_result.json \
  --min-overall-accuracy 0.55 \
  --min-worst-category-accuracy 0.35 \
  --min-samples-per-category 20
```

返回码：

1. `0`：通过。
2. `2`：未通过。

## 推荐执行顺序

1. 下载数据。
2. 预处理并切分。
3. 推荐路径：执行 `register_dataset_in_ms_swift.py`。
4. 推荐路径：执行 `train_ms_swift_lora.sh`。
5. 兼容旧路径：拉取 Qwen 仓库 + 注册数据集别名 + 执行 `train_qwenvl_lora.sh`。
6. 推理。
7. 评测。
8. 门禁判定。
