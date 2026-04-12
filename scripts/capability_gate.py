#!/usr/bin/env python3
"""Apply go/no-go capability gate from evaluation report."""

# 中文说明：
# - 功能：根据评测报告做门禁判定（通过/不通过）。
# - 判定项：overall_accuracy + worst_category_accuracy（可配置样本下限）。
# - 返回码：0 通过，2 不通过。

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply go/no-go gate from evaluation report. / 根据评测报告执行通过或不通过门禁。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--report",
        required=True,
        help="Path to evaluation report JSON / 评测报告 JSON 路径",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to gate result JSON / 门禁结果 JSON 输出路径",
    )
    parser.add_argument(
        "--min-overall-accuracy",
        type=float,
        default=0.55,
        help="Minimum overall accuracy threshold / overall accuracy 最低阈值",
    )
    parser.add_argument(
        "--min-worst-category-accuracy",
        type=float,
        default=0.35,
        help="Minimum worst-category accuracy threshold / 最差类别准确率最低阈值",
    )
    parser.add_argument(
        "--min-samples-per-category",
        type=int,
        default=20,
        help="Minimum samples required to include a category in worst-category check / 最差类别检查时每类最低样本数",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = Path(args.report).resolve()
    output_path = Path(args.output).resolve()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    by_category = report.get("by_category", {})

    # 只统计样本量达到阈值的类别，避免小样本类别干扰
    eligible = []
    for category, item in by_category.items():
        if int(item.get("total", 0)) >= args.min_samples_per_category:
            eligible.append((category, float(item.get("accuracy", 0.0))))

    worst_category = None
    worst_acc = None
    if eligible:
        worst_category, worst_acc = min(eligible, key=lambda x: x[1])

    checks = {
        "overall_accuracy": {
            "threshold": args.min_overall_accuracy,
            "value": float(report.get("overall_accuracy", 0.0)),
        },
        "worst_category_accuracy": {
            "threshold": args.min_worst_category_accuracy,
            "value": worst_acc,
            "category": worst_category,
            "applied": worst_acc is not None,
        },
    }

    reasons = []
    passed = True

    if checks["overall_accuracy"]["value"] < checks["overall_accuracy"]["threshold"]:
        passed = False
        reasons.append(
            "overall_accuracy below threshold: "
            f"{checks['overall_accuracy']['value']:.4f} < {checks['overall_accuracy']['threshold']:.4f}"
        )

    if worst_acc is not None and worst_acc < args.min_worst_category_accuracy:
        passed = False
        reasons.append(
            "worst_category_accuracy below threshold: "
            f"{worst_acc:.4f} < {args.min_worst_category_accuracy:.4f} ({worst_category})"
        )

    if not eligible:
        reasons.append("No category met min-samples-per-category; worst-category gate skipped")

    result = {
        "passed": passed,
        "checks": checks,
        "reasons": reasons,
        "source_report": str(report_path),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    sys.exit(0 if passed else 2)


if __name__ == "__main__":
    main()
