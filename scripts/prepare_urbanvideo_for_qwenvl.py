#!/usr/bin/env python3
"""Convert UrbanVideo-Bench MCQ data into QwenVL finetune JSONL format."""

# 中文说明：
# - 功能：把 parquet + videos 转成 QwenVL 训练格式 JSONL。
# - 关键点：按 video_id 切分 train/val/test，避免同视频泄漏。
# - 输出：train/val/test + ground_truth_* + prepare_summary.json。

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert UrbanVideo-Bench MCQ data into QwenVL finetune JSONL format. / 将 UrbanVideo-Bench 的 MCQ 数据转换为 QwenVL 微调 JSONL 格式。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw-root",
        default="data/raw/urbanvideo_bench",
        help="Root directory of raw dataset / 原始数据根目录",
    )
    parser.add_argument(
        "--videos-dirname",
        default="videos",
        help="Video folder name under raw-root / raw-root 下的视频目录名",
    )
    parser.add_argument(
        "--parquet-path",
        default=None,
        help="Optional explicit parquet file path / 可选：显式指定 parquet 文件路径",
    )
    parser.add_argument(
        "--output-root",
        default="data/processed/urbanvideo_bench",
        help="Output directory for processed files / 处理后文件输出目录",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio / 训练集比例",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio / 验证集比例",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio / 测试集比例",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split / 切分随机种子",
    )
    parser.add_argument(
        "--prompt-suffix",
        default="Reply with only one option letter (A/B/C/D), no explanation.",
        help="Suffix added to each question prompt / 每条问题后追加的提示语",
    )
    parser.add_argument(
        "--skip-missing-videos",
        action="store_true",
        help="Skip samples whose video file cannot be resolved / 跳过无法定位视频文件的样本",
    )
    return parser.parse_args()


def pick_column(columns: list[str], candidates: list[str], required: bool = True) -> str | None:
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise ValueError(f"None of candidate columns found: {candidates}. Available: {columns}")
    return None


def find_parquet_files(raw_root: Path, explicit_path: str | None) -> list[Path]:
    if explicit_path:
        path = Path(explicit_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {path}")
        return [path]

    candidates = sorted(raw_root.rglob("*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No parquet files found under {raw_root}")

    # 优先选择带 mcq 关键字的数据文件
    mcq = [p for p in candidates if "mcq" in p.name.lower()]
    return mcq if mcq else candidates


def normalize_letter(text: Any) -> str | None:
    if text is None:
        return None
    value = str(text).strip().upper()
    if not value:
        return None

    patterns = [
        r"\(([A-D])\)",
        r"\b([A-D])\b",
        r"OPTION\s*([A-D])",
        r"ANSWER\s*[:：]?\s*([A-D])",
    ]
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            return match.group(1)
    return None


def build_video_index(videos_root: Path) -> dict[str, Path]:
    if not videos_root.exists():
        raise FileNotFoundError(f"Videos folder not found: {videos_root}")

    # 建立视频索引：文件名和去扩展名都可检索
    index: dict[str, Path] = {}
    for path in videos_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        name_key = path.name.lower()
        stem_key = path.stem.lower()
        index[name_key] = path
        index[stem_key] = path
    return index


def resolve_video_path(video_id: Any, video_index: dict[str, Path]) -> Path | None:
    key = str(video_id).strip()
    if not key:
        return None

    key_lower = key.lower()
    candidates = [
        key_lower,
        Path(key_lower).name,
        Path(key_lower).stem,
    ]
    for ext in VIDEO_EXTENSIONS:
        candidates.append(f"{Path(key_lower).stem}{ext}")

    seen = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        if cand in video_index:
            return video_index[cand]
    return None


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_by_video(
    rows: list[dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"train+val+test must equal 1.0, got {total_ratio}")

    # 按 video_id 分组，确保同一视频只会进入一个集合
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["video_id"]].append(row)

    video_ids = list(groups.keys())
    random.Random(seed).shuffle(video_ids)

    n = len(video_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_vids = set(video_ids[:n_train])
    val_vids = set(video_ids[n_train : n_train + n_val])

    train_rows, val_rows, test_rows = [], [], []
    for video_id, samples in groups.items():
        if video_id in train_vids:
            train_rows.extend(samples)
        elif video_id in val_vids:
            val_rows.extend(samples)
        else:
            test_rows.extend(samples)

    return train_rows, val_rows, test_rows


def build_ground_truth(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gt = []
    for row in rows:
        gt.append(
            {
                "id": row["id"],
                "answer": row["raw_answer"],
                "answer_letter": row.get("answer_letter"),
                "question_category": row.get("question_category"),
            }
        )
    return gt


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root).resolve()
    output_root = Path(args.output_root).resolve()
    videos_root = raw_root / args.videos_dirname

    parquet_files = find_parquet_files(raw_root, args.parquet_path)
    frames = [pd.read_parquet(path) for path in parquet_files]
    frame = pd.concat(frames, ignore_index=True)
    columns = list(frame.columns)

    # 自动识别列名，兼容大小写或命名差异
    question_col = pick_column(columns, ["question", "Question"])
    answer_col = pick_column(columns, ["answer", "Answer", "label"])
    video_col = pick_column(columns, ["video_id", "video", "video_name"])
    qid_col = pick_column(columns, ["Question_id", "question_id", "id"], required=False)
    category_col = pick_column(columns, ["question_category", "category", "task_type"], required=False)

    video_index = build_video_index(videos_root)

    rows: list[dict[str, Any]] = []
    missing_video = 0

    for idx, row in frame.iterrows():
        video_id = str(row[video_col]).strip()
        video_path = resolve_video_path(video_id, video_index)
        if video_path is None:
            missing_video += 1
            if args.skip_missing_videos:
                continue
            raise FileNotFoundError(f"Cannot resolve video for video_id={video_id}")

        question_text = str(row[question_col]).strip()
        raw_answer = str(row[answer_col]).strip()
        answer_letter = normalize_letter(raw_answer)
        assistant_value = f"Answer: ({answer_letter})" if answer_letter else raw_answer
        prompt = f"{question_text}\n\nInstruction: {args.prompt_suffix}".strip()

        sample_id = str(row[qid_col]) if qid_col else str(idx)
        category = str(row[category_col]) if category_col else "unknown"
        rel_video_path = video_path.relative_to(raw_root).as_posix()

        # 生成 QwenVL 对话格式：human 提问 + gpt 标注答案
        rows.append(
            {
                "id": sample_id,
                "video_id": video_id,
                "question_category": category,
                "raw_answer": raw_answer,
                "answer_letter": answer_letter,
                "video": rel_video_path,
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<video>\n{prompt}",
                    },
                    {
                        "from": "gpt",
                        "value": assistant_value,
                    },
                ],
            }
        )

    train_rows, val_rows, test_rows = split_by_video(
        rows=rows,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    train_path = output_root / "train.jsonl"
    val_path = output_root / "val.jsonl"
    test_path = output_root / "test.jsonl"

    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    write_jsonl(test_path, test_rows)

    write_jsonl(output_root / "ground_truth_train.jsonl", build_ground_truth(train_rows))
    write_jsonl(output_root / "ground_truth_val.jsonl", build_ground_truth(val_rows))
    write_jsonl(output_root / "ground_truth_test.jsonl", build_ground_truth(test_rows))

    summary = {
        "raw_root": str(raw_root),
        "videos_root": str(videos_root),
        "parquet_files": [str(p) for p in parquet_files],
        "columns": columns,
        "question_column": question_col,
        "answer_column": answer_col,
        "video_column": video_col,
        "question_id_column": qid_col,
        "category_column": category_col,
        "total_samples": len(rows),
        "missing_video_count": missing_video,
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "test_samples": len(test_rows),
        "train_output": str(train_path),
        "val_output": str(val_path),
        "test_output": str(test_path),
    }
    summary_path = output_root / "prepare_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("UrbanVideo-Bench conversion done.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
