#!/usr/bin/env python3
"""Prepare UrbanVideo-Bench JSONL for ms-swift and generate dataset_info.json."""

# 中文说明：
# - 功能：把现有 QwenVL JSONL 转成 ms-swift 标准格式，并生成 dataset_info.json。
# - 关键点：将每条样本的视频路径转为绝对路径，避免训练时因工作目录不同导致找不到视频。
# - 输出：train/val/test 的 ms-swift JSONL + dataset_info.json。

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert QwenVL JSONL to ms-swift format and register dataset_info.json. "
            "/ 转换 QwenVL JSONL 为 ms-swift 格式并生成 dataset_info.json。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        default="data/raw/urbanvideo_bench",
        help="Root path used to resolve relative video path / 用于解析相对视频路径的数据根目录",
    )
    parser.add_argument(
        "--train-jsonl",
        default="data/processed/urbanvideo_bench/train.jsonl",
        help="Source train JSONL in QwenVL format / QwenVL 格式训练集 JSONL",
    )
    parser.add_argument(
        "--val-jsonl",
        default="data/processed/urbanvideo_bench/val.jsonl",
        help="Source val JSONL in QwenVL format / QwenVL 格式验证集 JSONL",
    )
    parser.add_argument(
        "--test-jsonl",
        default="data/processed/urbanvideo_bench/test.jsonl",
        help="Source test JSONL in QwenVL format / QwenVL 格式测试集 JSONL",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/urbanvideo_bench/ms_swift",
        help="Output directory for converted ms-swift JSONL / 转换后 ms-swift JSONL 输出目录",
    )
    parser.add_argument(
        "--dataset-info-path",
        default="outputs/ms_swift/dataset_info.json",
        help="Output path of custom dataset_info.json / 自定义 dataset_info.json 输出路径",
    )
    parser.add_argument(
        "--dataset-prefix",
        default="urbanvideo",
        help="Dataset alias prefix in dataset_info.json / dataset_info.json 中的数据集别名前缀",
    )
    parser.add_argument(
        "--skip-missing-videos",
        action="store_true",
        help="Skip samples whose videos cannot be resolved / 跳过无法定位视频文件的样本",
    )
    parser.add_argument(
        "--keep-extra-fields",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep extra metadata fields (id/category/etc.) / 保留额外元字段（id/category 等）",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_role(value: Any) -> str:
    role = str(value).strip().lower()
    mapping = {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
        "bot": "assistant",
        "system": "system",
    }
    return mapping.get(role, role if role else "user")


def convert_conversations_to_messages(conversations: Any) -> list[dict[str, str]]:
    if not isinstance(conversations, list) or not conversations:
        raise ValueError("Invalid conversations field")

    messages: list[dict[str, str]] = []
    for turn in conversations:
        if not isinstance(turn, dict):
            continue

        content = turn.get("value")
        if content is None:
            content = turn.get("content")
        if content is None:
            continue

        role = turn.get("from")
        if role is None:
            role = turn.get("role")

        messages.append(
            {
                "role": normalize_role(role),
                "content": str(content),
            }
        )

    if not messages:
        raise ValueError("Converted messages is empty")
    return messages


def resolve_video_path(row: dict[str, Any], data_root: Path) -> Path:
    if "videos" in row and isinstance(row["videos"], list) and row["videos"]:
        candidate = str(row["videos"][0])
    elif "video" in row:
        candidate = str(row["video"])
    else:
        raise ValueError("Missing video/videos field")

    video_path = Path(candidate)
    if not video_path.is_absolute():
        video_path = (data_root / video_path).resolve()
    else:
        video_path = video_path.resolve()

    return video_path


def convert_split(
    split_name: str,
    source_jsonl: Path,
    output_jsonl: Path,
    data_root: Path,
    keep_extra_fields: bool,
    skip_missing_videos: bool,
) -> tuple[int, int]:
    rows = read_jsonl(source_jsonl)
    converted: list[dict[str, Any]] = []
    skipped_missing_video = 0

    keep_keys = [
        "id",
        "video_id",
        "question_category",
        "allowed_letters",
        "answer_letter",
        "raw_answer",
    ]

    for row in rows:
        messages = convert_conversations_to_messages(row.get("conversations"))
        video_path = resolve_video_path(row, data_root=data_root)

        if not video_path.exists():
            if skip_missing_videos:
                skipped_missing_video += 1
                continue
            raise FileNotFoundError(
                f"[{split_name}] video file not found: {video_path} (id={row.get('id')})"
            )

        item: dict[str, Any] = {
            "messages": messages,
            "videos": [video_path.as_posix()],
        }

        if keep_extra_fields:
            for key in keep_keys:
                if key in row:
                    item[key] = row[key]

        converted.append(item)

    write_jsonl(output_jsonl, converted)
    return len(converted), skipped_missing_video


def make_dataset_entry(dataset_name: str, dataset_path: Path) -> dict[str, Any]:
    return {
        "dataset_name": dataset_name,
        "dataset_path": dataset_path.as_posix(),
        "split": ["train"],
    }


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    dataset_info_path = Path(args.dataset_info_path).resolve()

    split_sources = {
        "train": Path(args.train_jsonl).resolve(),
        "val": Path(args.val_jsonl).resolve(),
        "test": Path(args.test_jsonl).resolve(),
    }

    for required_split in ["train", "val"]:
        if not split_sources[required_split].exists():
            raise FileNotFoundError(f"Source {required_split} JSONL not found: {split_sources[required_split]}")

    stats: dict[str, dict[str, int | str]] = {}
    dataset_info: list[dict[str, Any]] = []

    for split, source_path in split_sources.items():
        if not source_path.exists():
            print(f"Skip {split}: source file not found -> {source_path}")
            continue

        output_jsonl = output_dir / f"{split}_ms_swift.jsonl"
        kept, skipped = convert_split(
            split_name=split,
            source_jsonl=source_path,
            output_jsonl=output_jsonl,
            data_root=data_root,
            keep_extra_fields=args.keep_extra_fields,
            skip_missing_videos=args.skip_missing_videos,
        )

        dataset_alias = f"{args.dataset_prefix}_{split}"
        dataset_info.append(make_dataset_entry(dataset_name=dataset_alias, dataset_path=output_jsonl))

        stats[split] = {
            "source": source_path.as_posix(),
            "output": output_jsonl.as_posix(),
            "kept_samples": kept,
            "skipped_missing_video": skipped,
            "dataset_alias": dataset_alias,
        }

    dataset_info_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_info_path.write_text(json.dumps(dataset_info, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "data_root": data_root.as_posix(),
        "dataset_info_path": dataset_info_path.as_posix(),
        "splits": stats,
    }

    print("ms-swift dataset conversion and registration done.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
