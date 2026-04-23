# UrbanVideo-Bench 视频QA项目中文说明（QwenVL 3.0/3.5）

本文档用于说明三件事：

1. 项目从下载到能力判定的完整流程。
2. 工程中关键文件的职责。
3. 推荐执行顺序与落地步骤。

脚本速查入口：scripts/README_zh.md（重点看每个脚本的输入、输出和命令示例）
云端手动命令手册（Qwen3.5-9B-GPTQ LoRA 冒烟）：CLOUD_LORA_SMOKE_QWEN3_5_9B_GPTQ_RUNBOOK.md

## 1. 项目目标

本项目面向赛题的视频问答（VideoQA）子任务，核心目标是：

1. 使用 Hugging Face 上的 UrbanVideo-Bench 数据。
2. 以上云 QwenVL 3.0 或 3.5（优先 7B Instruct）为基座做 LoRA 微调。
3. 用 MCQ 准确率评估模型能力。
4. 基于阈值做 go/no-go（是否达标）判定。

## 2. 一图看完整流程

赛题文档理解
-> 数据下载（Hugging Face）
-> 数据转换（QwenVL JSONL）
-> 训练/验证/测试切分（按 video_id 防泄漏）
-> Qwen 官方框架注册数据集
-> 云端 LoRA 训练
-> 测试集推理
-> 准确率评估（overall + 分类别）
-> 能力门禁判定（pass/fail）

## 3. 当前目录结构（核心）

- configs：流程参数样例。
- scripts：下载、预处理、训练、推理、评测脚本。
- data：原始数据和处理后数据。
- outputs：预测结果、评估报告、模型输出。
- third_party：Qwen 官方仓库。

## 4. 端到端流程详解（执行步骤）

### 步骤 0：准备环境

本地（用于数据处理和评估）：

```bash
pip install -r requirements.txt
```

云端（用于训练和推理）建议：

1. Linux + CUDA 环境。
2. 安装 PyTorch、Transformers（新版本）、qwen-vl-utils[decord]、peft、accelerate。
3. 如果显存紧张，先试 3B 或减小 batch / frame 数。

云端模型建议：

1. 优先：Qwen/Qwen3.5-VL-7B-Instruct（精度优先）。
2. 备选：Qwen/Qwen3-VL-7B-Instruct（兼容性优先）。
3. 本地冒烟仍可用 2.5/3B/int4，正式指标以云端 3.0/3.5 结果为准。

### 步骤 1：下载 UrbanVideo-Bench

```bash
python scripts/download_urbanvideo_bench.py \
  --dataset-id EmbodiedCity/UrbanVideo-Bench \
   --local-dir data/raw/urbanvideo_bench \
   --hf-endpoint https://hf-mirror.com \
   --sample-records 200
```

输入：Hugging Face 数据集。
输出：data/raw/urbanvideo_bench + download_manifest.json。
通过标准：manifest 里有 parquet 和 videos 文件。

说明：

1. 默认已使用 `hf-mirror.com` 作为下载端点。
2. 默认会按 `question_category` 分层随机采样 200 条记录，再下载对应视频，适合云端 LoRA 冒烟测试。
3. 如需全量下载，可增加参数 `--full-download`。

### 步骤 2：转换为 QwenVL 训练格式 + 切分数据

```bash
python scripts/prepare_urbanvideo_for_qwenvl.py \
  --raw-root data/raw/urbanvideo_bench \
  --output-root data/processed/urbanvideo_bench \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1
```

输入：parquet + videos。
输出：

1. train.jsonl / val.jsonl / test.jsonl。
2. ground_truth_train.jsonl / ground_truth_val.jsonl / ground_truth_test.jsonl。
3. prepare_summary.json。

关键点：按 video_id 切分，避免同一视频泄漏到不同集合。

### 步骤 3：同步 Qwen 官方训练仓库

Windows PowerShell：

```powershell
./scripts/setup_qwenvl_repo.ps1 -RepoDir third_party/Qwen3.5-VL -RepoUrl https://github.com/QwenLM/Qwen3.5-VL.git
```

备选（3.0）：

```powershell
./scripts/setup_qwenvl_repo.ps1 -RepoDir third_party/Qwen3-VL -RepoUrl https://github.com/QwenLM/Qwen3-VL.git
```

作用：克隆或更新第三方 QwenVL 仓库（推荐 3.0/3.5）。

### 步骤 4（推荐）：生成 ms-swift 数据集映射

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

作用：把原有 QwenVL JSONL 转为 ms-swift 可直接训练的格式，并生成 `dataset_info.json`。

默认会注册 3 个别名：

1. `urbanvideo_train`
2. `urbanvideo_val`
3. `urbanvideo_test`

### 步骤 5（推荐）：云端 LoRA 训练（ms-swift）

```bash
pip install -U ms-swift

bash scripts/train_ms_swift_lora.sh
```

可覆盖参数示例：

```bash
MODEL_NAME_OR_PATH=Qwen/Qwen3.5-VL-7B-Instruct \
DATASET_INFO_PATH=outputs/ms_swift/dataset_info.json \
TRAIN_DATASET=urbanvideo_train \
VAL_DATASET=urbanvideo_val \
DEEPSPEED=zero2 \
bash scripts/train_ms_swift_lora.sh
```

输出：`outputs/checkpoints/ms_swift_qwen2_5_vl_7b_lora`（建议按模型版本区分命名）。

### 步骤 4B（兼容旧链路）：把自定义数据集注册到 Qwen 框架

```bash
python scripts/register_dataset_in_qwenvl.py \
   --qwenvl-data-init third_party/<QWENVL_REPO>/qwen-vl-finetune/qwenvl/data/__init__.py \
  --data-path data/raw/urbanvideo_bench \
  --dataset urbanvideo_train=data/processed/urbanvideo_bench/train.jsonl \
  --dataset urbanvideo_val=data/processed/urbanvideo_bench/val.jsonl
```

作用：把训练/验证别名写入 data_dict，后续训练参数可直接用别名。

说明：`<QWENVL_REPO>` 可选 `Qwen3-VL` 或 `Qwen3.5-VL`。

### 步骤 5B（兼容旧链路）：云端 LoRA 训练（Qwen 官方）

```bash
bash scripts/train_qwenvl_lora.sh
```

可覆盖参数示例：

```bash
MODEL_NAME_OR_PATH=Qwen/Qwen3.5-VL-7B-Instruct \
DATASET_USE=urbanvideo_train%100 \
NPROC_PER_NODE=2 \
bash scripts/train_qwenvl_lora.sh
```

可替换为：`MODEL_NAME_OR_PATH=Qwen/Qwen3-VL-7B-Instruct`。

输出：outputs/checkpoints/qwen_vl_7b_lora（建议自定义命名，区分 3.0/3.5）。

### 步骤 6：测试集推理

```bash
python scripts/infer_qwen2_5_vl_mcq.py \
  --model-name-or-path <你的模型或检查点路径> \
  --input-jsonl data/processed/urbanvideo_bench/test.jsonl \
  --data-root data/raw/urbanvideo_bench \
  --output-jsonl outputs/predictions/test_predictions.jsonl
```

高吞吐（推荐比赛冲榜时使用，Linux + GPU）：

```bash
pip install vllm outlines qwen-vl-utils[decord]

python scripts/infer_qwen2_5_vl_mcq.py \
   --backend vllm \
   --model-name-or-path <你的模型或检查点路径> \
   --input-jsonl data/processed/urbanvideo_bench/test.jsonl \
   --data-root data/raw/urbanvideo_bench \
   --output-jsonl outputs/predictions/test_predictions_vllm.jsonl \
   --batch-size 16 \
   --constrained-decoding
```

说明：

1. `--backend transformers` 为默认模式，兼容性更高。
2. `--backend vllm` 使用连续批处理，通常能显著提升吞吐。
3. `--constrained-decoding` 会把输出限制在题目允许选项（如 A/B/C/D/E）内，减少解析失败。

输出：test_predictions.jsonl（包含 prediction 与 prediction_letter）。

### 步骤 7：计算准确率

```bash
python scripts/eval_mcq_accuracy.py \
  --ground-truth data/processed/urbanvideo_bench/ground_truth_test.jsonl \
  --predictions outputs/predictions/test_predictions.jsonl \
  --report-path outputs/reports/test_eval_report.json
```

输出指标：

1. overall_accuracy。
2. by_category（每类样本数、正确数、准确率）。
3. 缺失预测与多余预测统计。

### 步骤 8：能力门禁（是否达标）

```bash
python scripts/capability_gate.py \
  --report outputs/reports/test_eval_report.json \
  --output outputs/reports/test_gate_result.json \
  --min-overall-accuracy 0.55 \
  --min-worst-category-accuracy 0.35 \
  --min-samples-per-category 20
```

返回码：

1. 0：通过。
2. 2：未通过。

## 5. 文件说明

### 根目录文件

1. .gitignore
   用途：忽略数据、输出、第三方仓库和临时文件，避免仓库污染。

2. requirements.txt
   用途：本地预处理与评测依赖。

3. README.md
   用途：英文快速执行指南。

4. README_CN.md
   用途：中文全流程说明（本文件）。

### 配置文件

1. configs/pipeline.example.yaml
   用途：样例参数集合（数据源、切分比例、训练别名、门禁阈值）。

### 脚本文件（scripts）

1. scripts/download_urbanvideo_bench.py
   作用：下载数据并生成 manifest。

2. scripts/prepare_urbanvideo_for_qwenvl.py
   作用：parquet + 视频转 QwenVL JSONL，并按 video_id 切分。

3. scripts/register_dataset_in_qwenvl.py
   作用：自动把你的 JSONL 数据注册进 Qwen 训练框架。

4. scripts/register_dataset_in_ms_swift.py
   作用：把 QwenVL JSONL 转成 ms-swift 训练格式并生成 dataset_info.json。

5. scripts/setup_qwenvl_repo.ps1
   作用：克隆或更新 QwenVL 官方仓库（推荐 3.0/3.5）。

6. scripts/train_qwenvl_lora.sh
   作用：云端 LoRA 训练启动脚本。

7. scripts/train_ms_swift_lora.sh
   作用：基于 ms-swift 的 LoRA 训练启动脚本（推荐）。

8. scripts/infer_qwen2_5_vl_mcq.py
   作用：对指定 split 执行推理并导出预测。

9. scripts/eval_mcq_accuracy.py
   作用：计算 overall 与分类型准确率。

10. scripts/capability_gate.py
   作用：按阈值给出 pass/fail 结论。

## 6. 建议执行顺序

### A. 先做一次小规模冒烟验证（建议今天完成）

1. 执行下载和预处理，确认数据链路无误。
2. 抽样少量 test 先跑推理，确认模型可出结果。
3. 跑 eval 和 gate，确认评测链路完整。

目标：先验证“流程能跑通”，再投入大规模训练。

### B. 云端正式训练（建议 1-2 天内完成首轮）

1. 优先走 ms-swift 链路（步骤 4-5）跑首轮 7B LoRA，保存 checkpoint。
2. 用同一套 test 流程生成首版准确率报告。
3. 记录训练配置和结果，建立对比基线。

### C. 第二轮调参（按效果迭代）

可优先调整：

1. 学习率和 epoch。
2. 视频抽帧参数（fps、max_frames）。
3. LoRA 参数（r、alpha、dropout）。
4. 数据比例（先 30%-50% 快速试验，再全量）。

### D. 交付前检查

1. 固定一版模型和配置。
2. 生成最终 test 预测文件。
3. 产出最终评估报告与门禁结果。
4. 归档命令、日志、配置，确保可复现。

## 7. 常见问题与排查

1. 报错 KeyError: qwen2_5_vl
   原因：transformers 版本太旧，或当前版本不支持所选 QwenVL。
   处理：升级到支持 QwenVL 3.0/3.5 的新版本。

2. 推理报视频找不到
   原因：data-root 与 JSONL 里的相对路径拼接不一致。
   处理：检查 data/raw/urbanvideo_bench 下视频路径和 JSONL 的 video 字段。

3. 训练显存不足
   处理：降低 batch、减少 frame、启用更小模型（如 3B）做快速验证。

4. 评估分数异常偏低
   处理：先检查 prediction_letter 提取是否正常，再看是否有 missing_prediction_count 偏高；可优先使用 `--constrained-decoding` 以减少无效输出。

## 8. 现在最推荐你执行的三步

1. 先跑步骤 1-2，拿到 data/processed/urbanvideo_bench 的完整输出。
2. 上云跑步骤 3-5，拿到第一版 checkpoint。
3. 运行步骤 6-8，得到第一版准确率和 gate 结果，再决定第二轮调参方向。
