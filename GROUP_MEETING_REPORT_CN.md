# UrbanVideo-Bench VideoQA 项目组会汇报（Qwen2.5-VL LoRA）

## 1. 项目目标

本项目面向赛题中的视频问答（VideoQA）多选题（MCQ）场景，目标是搭建一条可复现的端到端流程：

1. 从 Hugging Face 下载 UrbanVideo-Bench 数据。
2. 将数据转换为 QwenVL 可训练格式并完成防泄漏切分。
3. 基于 Qwen2.5-VL 做 LoRA 微调，得到首轮可用模型。
4. 在测试集执行推理、评测，并给出 go/no-go 能力判定。

## 2. 技术路线

1. 数据集：UrbanVideo-Bench（视频+题目+选项+答案）。
2. 模型：Qwen2.5-VL-7B-Instruct（低预算可先用 3B 验证）。
3. 训练方式：LoRA 微调。
4. 评测指标：overall accuracy + 分类别 accuracy。
5. 门禁规则：overall 与 worst-category 达到阈值即通过。

## 3. 端到端流程与文件映射

| 阶段 | 做的事 | 主要产出 | 对应文件 |
|---|---|---|---|
| 需求与指标定义 | 明确任务、指标、验收标准 | 流程与口径统一 | README_CN.md, README.md |
| 数据下载 | 从 HF 拉取 UrbanVideo-Bench | 原始数据目录、manifest | scripts/download_urbanvideo_bench.py |
| 数据转换与切分 | 转换为 QwenVL JSONL，按 video_id 切分 | train/val/test JSONL、ground truth | scripts/prepare_urbanvideo_for_qwenvl.py |
| 框架准备 | 克隆或更新 Qwen 官方仓库 | 可训练代码环境 | scripts/setup_qwenvl_repo.ps1 |
| 数据注册 | 把自定义数据别名写入 Qwen data_dict | 可直接按别名训练 | scripts/register_dataset_in_qwenvl.py |
| LoRA 训练 | 启动训练并保存权重 | checkpoint 目录 | scripts/train_qwenvl_lora.sh |
| 测试推理 | 生成测试集预测 | prediction JSONL | scripts/infer_qwen2_5_vl_mcq.py |
| 评测与门禁 | 计算准确率并做 pass/fail | report JSON、gate JSON | scripts/eval_mcq_accuracy.py, scripts/capability_gate.py |
| 文档与交付 | 汇总执行方式、命令、口径 | 中文总览和速查文档 | README_CN.md, scripts/README_zh.md |

## 4. 当前完成情况（可汇报）

1. 已完成端到端脚本骨架：下载、转换、注册、训练、推理、评测、门禁全流程齐全。
2. 已完成中文文档体系：项目中文总览 + 脚本速查文档。
3. 已完成脚本可读性增强：核心脚本均有中文注释。
4. 当前阶段可直接进入首轮云端训练，生成第一版基线报告。

## 5. 预算估算（建议汇报口径）

估算公式：

总成本 ≈ GPU单价 × GPU小时 + 存储/流量 + 20%-30%重跑冗余

| 档位 | 推荐配置 | 预计时长 | 价格区间 | 预计总成本 | 适用场景 |
|---|---|---|---|---|---|
| 低预算验证档 | 1 x RTX 4090 24G | 12-20 小时 | 5-10 元/小时 | 80-250 元 | 跑通流程、拿首版结果 |
| 推荐主线档 | 1 x A6000 48G 或 2 x 4090 | 20-40 小时 | 10-22 元/小时 | 300-900 元 | 首轮正式训练+1次调参 |
| 稳妥冲刺档 | 1 x A100 80G | 30-60 小时 | 20-40 元/小时 | 800-2500 元 | 多轮调参、冲刺成绩 |

建议组会申请预算：

1. 首批 400-800 元用于两轮实验（主线方案）。
2. 额外预留 200-500 元用于重跑与故障缓冲。

## 6. 平台推荐与选择理由

| 平台类型 | 示例平台 | 优势 | 风险 | 适用阶段 |
|---|---|---|---|---|
| 共享算力平台 | AutoDL, Vast.ai, RunPod | 性价比高、按小时弹性 | 稳定性与排队存在波动 | 前期验证、低成本试验 |
| 云厂商训练平台 | 阿里云 PAI, 腾讯云 TI, 华为云 ModelArts | 稳定、运维支持好、交付风险低 | 单价更高 | 正式训练与交付阶段 |

推荐策略：

1. 第一轮先上 4090 档拿到可汇报结果。
2. 若显存或稳定性不足，再切到 A6000/A100 档。
3. 以“先低成本验证，再按结果升级算力”作为组会共识。

## 7. 本次组会建议决策项

1. 是否批准首批 400-800 元训练预算。
2. 是否采用“先 4090 验证，再按效果升级”的两阶段算力策略。
3. 是否以本周产出“首版评测报告+门禁结果”为阶段里程碑。

## 8. 下周执行计划（可直接汇报）

1. Day 1：完成云端环境就绪与数据链路冒烟。
2. Day 2-3：完成首轮 LoRA 训练并产出 checkpoint。
3. Day 4：完成测试集推理与 accuracy 报告。
4. Day 5：完成门禁判定与参数复盘，确定第二轮调参方向。

## 9. 风险与应对

1. 显存不足。
应对：降低 batch、减少帧数、先切 3B 验证。

2. 视频路径不一致导致推理失败。
应对：统一 data-root 与 JSONL 相对路径规则。

3. 依赖版本不兼容。
应对：固定 transformers 与 qwen-vl-utils 版本，先做小规模冒烟。

---

以上报告可直接用于组会讲解，结构为：目标 -> 路线 -> 流程 -> 文件映射 -> 预算 -> 平台 -> 决策项 -> 周计划。
