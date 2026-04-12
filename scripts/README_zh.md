# scripts 目录中文说明

这个文档是给你“直接看脚本就能跑”的速查版，重点看每个脚本的输入、输出、常见命令。

## 1) download_urbanvideo_bench.py

作用：从 Hugging Face 下载 UrbanVideo-Bench 到本地。

常用命令：

```bash
python scripts/download_urbanvideo_bench.py \
  --dataset-id EmbodiedCity/UrbanVideo-Bench \
  --local-dir data/raw/urbanvideo_bench
```

输出：

1. 数据文件（parquet/videos）。
2. `download_manifest.json`（下载清单）。

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

作用：拉取或更新 Qwen2.5-VL 官方仓库。

常用命令：

```powershell
powershell ./scripts/setup_qwenvl_repo.ps1
powershell ./scripts/setup_qwenvl_repo.ps1 -Pull
```

## 4) register_dataset_in_qwenvl.py

作用：把你处理后的 JSONL 注册成 Qwen 框架可识别的数据集别名。

常用命令：

```bash
python scripts/register_dataset_in_qwenvl.py \
  --qwenvl-data-init third_party/Qwen2.5-VL/qwen-vl-finetune/qwenvl/data/__init__.py \
  --data-path data/raw/urbanvideo_bench \
  --dataset urbanvideo_train=data/processed/urbanvideo_bench/train.jsonl \
  --dataset urbanvideo_val=data/processed/urbanvideo_bench/val.jsonl
```

## 5) train_qwenvl_lora.sh

作用：云端启动 LoRA 训练（默认低预算参数）。

常用命令：

```bash
bash scripts/train_qwenvl_lora.sh
```

带覆盖参数：

```bash
MODEL_NAME_OR_PATH=Qwen/Qwen2.5-VL-7B-Instruct \
DATASET_USE=urbanvideo_train%100 \
NPROC_PER_NODE=2 \
bash scripts/train_qwenvl_lora.sh
```

输出：`outputs/checkpoints/qwen2_5_vl_7b_lora`。

## 6) infer_qwen2_5_vl_mcq.py

作用：在某个 split（通常 test）上跑推理，导出预测。

常用命令：

```bash
python scripts/infer_qwen2_5_vl_mcq.py \
  --model-name-or-path <模型路径> \
  --input-jsonl data/processed/urbanvideo_bench/test.jsonl \
  --data-root data/raw/urbanvideo_bench \
  --output-jsonl outputs/predictions/test_predictions.jsonl
```

## 7) eval_mcq_accuracy.py

作用：计算 MCQ 准确率（总分 + 分类别）。

常用命令：

```bash
python scripts/eval_mcq_accuracy.py \
  --ground-truth data/processed/urbanvideo_bench/ground_truth_test.jsonl \
  --predictions outputs/predictions/test_predictions.jsonl \
  --report-path outputs/reports/test_eval_report.json
```

## 8) capability_gate.py

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
3. 拉取 Qwen 仓库。
4. 注册数据集别名。
5. 云端训练。
6. 推理。
7. 评测。
8. 门禁判定。
