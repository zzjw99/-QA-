#!/usr/bin/env python3
"""Evaluate MCQ predictions with overall and per-category accuracy."""

# 中文说明：
# - 功能：按 id 对齐预测与真值，计算 overall + 分类别准确率。
# - 兼容：支持多种 id 字段名和预测字段名。
# - 输出：report json，可直接用于 capability_gate.py。

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MCQ predictions with category breakdown. / 按类别统计并评测 MCQ 预测结果。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ground-truth",
        required=True,
        help="Ground-truth JSONL file / 真值标注 JSONL 文件",
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Prediction JSONL file / 预测结果 JSONL 文件",
    )
    parser.add_argument(
        "--report-path",
        required=True,
        help="Output report JSON path / 评测报告 JSON 输出路径",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_letter(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None

    patterns = [r"\(([A-D])\)", r"\b([A-D])\b", r"OPTION\s*([A-D])", r"ANSWER\s*[:：]?\s*([A-D])"]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


def detect_id(row: dict[str, Any]) -> str | None:
    for key in ("id", "question_id", "Question_id", "qid"):
        if key in row:
            return str(row[key])
    return None


def detect_prediction(row: dict[str, Any]) -> str:
    for key in ("prediction", "pred", "answer", "output", "response", "text"):
        if key in row:
            return str(row[key])
    raise KeyError(f"Cannot find prediction field in row keys: {list(row.keys())}")


def evaluate(
    gt_rows: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    # 建立真值索引（id -> row）
    gt_by_id: dict[str, dict[str, Any]] = {}
    for row in gt_rows:
        item_id = detect_id(row)
        if item_id is None:
            raise KeyError(f"Ground truth row missing id: {row}")
        gt_by_id[item_id] = row

    # 建立预测索引（id -> row）
    pred_by_id: dict[str, dict[str, Any]] = {}
    for row in pred_rows:
        item_id = detect_id(row)
        if item_id is None:
            continue
        pred_by_id[item_id] = row

    total = len(gt_by_id)
    correct = 0
    scored = 0
    missing = 0

    cat_total = defaultdict(int)
    cat_correct = defaultdict(int)

    for item_id, gt in gt_by_id.items():
        category = str(gt.get("question_category", "unknown"))
        cat_total[category] += 1

        pred = pred_by_id.get(item_id)
        if pred is None:
            missing += 1
            continue

        scored += 1
        gt_letter = normalize_letter(gt.get("answer_letter")) or normalize_letter(gt.get("answer"))
        pred_text = detect_prediction(pred)
        pred_letter = normalize_letter(pred.get("prediction_letter")) or normalize_letter(pred_text)

        # 优先比较选项字母；若无法解析则退化为文本精确匹配
        if gt_letter and pred_letter:
            is_correct = gt_letter == pred_letter
        else:
            is_correct = str(gt.get("answer", "")).strip().lower() == pred_text.strip().lower()

        if is_correct:
            correct += 1
            cat_correct[category] += 1

    by_category = {}
    for category, cat_n in sorted(cat_total.items()):
        cat_c = cat_correct.get(category, 0)
        by_category[category] = {
            "total": cat_n,
            "correct": cat_c,
            "accuracy": (cat_c / cat_n) if cat_n > 0 else 0.0,
        }

    extra_predictions = max(0, len(pred_by_id) - len(gt_by_id))
    report = {
        "total_ground_truth": total,
        "total_predictions": len(pred_by_id),
        "scored_predictions": scored,
        "correct": correct,
        "overall_accuracy": (correct / total) if total > 0 else 0.0,
        "missing_prediction_count": missing,
        "extra_prediction_count": extra_predictions,
        "by_category": by_category,
    }
    return report


def main() -> None:
    args = parse_args()
    gt_path = Path(args.ground_truth).resolve()
    pred_path = Path(args.predictions).resolve()
    report_path = Path(args.report_path).resolve()

    gt_rows = read_jsonl(gt_path)
    pred_rows = read_jsonl(pred_path)
    report = evaluate(gt_rows, pred_rows)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()
