#!/usr/bin/env python3
"""Download UrbanVideo-Bench dataset files from Hugging Face Hub."""

# 中文说明：
# - 功能：从 Hugging Face 拉取赛题数据到本地目录。
# - 输入：dataset-id、revision、allow-patterns。
# - 输出：本地数据文件 + download_manifest.json（用于核对下载内容）。

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download


VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download UrbanVideo-Bench dataset files from Hugging Face Hub. / 从 Hugging Face 下载 UrbanVideo-Bench 数据集文件。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-id",
        default="EmbodiedCity/UrbanVideo-Bench",
        help="Hugging Face dataset id / Hugging Face 数据集 ID",
    )
    parser.add_argument(
        "--local-dir",
        default="data/raw/urbanvideo_bench",
        help="Local output directory / 本地保存目录",
    )
    parser.add_argument(
        "--allow-patterns",
        default="*.parquet,videos/**,README.md,*.json,*.jsonl",
        help="Comma-separated allow patterns for snapshot_download / 逗号分隔的下载白名单模式",
    )
    parser.add_argument(
        "--hf-endpoint",
        default="https://hf-mirror.com",
        help="Hugging Face endpoint or mirror URL / Hugging Face 访问端点（可设为镜像）",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Dataset revision/tag/commit / 数据版本（分支、标签或提交）",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face token for private/gated assets / 私有或受限资源的可选访问令牌",
    )
    parser.add_argument(
        "--sample-records",
        type=int,
        default=200,
        help="Number of records to sample for smoke run (None means no record cap) / 冒烟版采样条数（None 表示不限制）",
    )
    parser.add_argument(
        "--sample-strategy",
        default="stratified",
        choices=["stratified", "random"],
        help="Sampling strategy for records / 记录采样策略",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Optional cap on sampled videos after record sampling / 可选：在记录采样后限制视频数",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed used when sampling videos / 采样视频时使用的随机种子",
    )
    parser.add_argument(
        "--full-download",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Download full dataset directly without sampling / 不采样，直接全量下载",
    )
    parser.add_argument(
        "--keep-full-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep full parquet metadata as backup metadata_full.parquet / 保留全量 parquet 备份",
    )
    return parser.parse_args()


def build_video_candidates(video_id: str) -> list[str]:
    value = str(video_id).strip().replace("\\", "/")
    if not value:
        return []

    # 兼容 video_id 可能带或不带 videos/ 前缀
    relative = value[7:] if value.startswith("videos/") else value

    candidates: list[str] = []
    if Path(relative).suffix:
        candidates.append(f"videos/{relative}")
    else:
        for ext in VIDEO_EXTENSIONS:
            candidates.append(f"videos/{relative}{ext}")

    unique = []
    seen = set()
    for item in candidates:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def resolve_parquet_path(local_dir: Path) -> Path:
    preferred = local_dir / "MCQ.parquet"
    if preferred.exists():
        return preferred

    candidates = sorted(local_dir.rglob("*.parquet"))
    if not candidates:
        raise FileNotFoundError("No parquet file found. Please ensure metadata is downloaded first.")
    return candidates[0]


def pick_video_column(columns: list[str]) -> str:
    lower_map = {col.lower(): col for col in columns}
    for candidate in ["video_id", "video", "video_name"]:
        if candidate in columns:
            return candidate
        if candidate in lower_map:
            return lower_map[candidate]
    raise KeyError(f"Cannot find video id column in parquet, available columns: {columns}")


def _category_distribution(frame: pd.DataFrame, category_col: str | None) -> dict[str, int]:
    if category_col is None or category_col not in frame.columns:
        return {}
    values = frame[category_col].fillna("unknown").astype(str).tolist()
    counter = Counter(values)
    return dict(sorted(counter.items(), key=lambda item: item[0]))


def _allocate_stratified_counts(
    category_sizes: dict[str, int],
    sample_records: int,
) -> dict[str, int]:
    total = sum(category_sizes.values())
    if total <= 0:
        return {}

    categories = list(category_sizes.keys())
    raw = {cat: category_sizes[cat] / total * sample_records for cat in categories}
    alloc = {cat: int(raw[cat]) for cat in categories}

    if sample_records >= len(categories):
        for cat in categories:
            if category_sizes[cat] > 0 and alloc[cat] == 0:
                alloc[cat] = 1

    for cat in categories:
        alloc[cat] = min(alloc[cat], category_sizes[cat])

    current = sum(alloc.values())

    if current > sample_records:
        reducible = [cat for cat in categories if alloc[cat] > 1]
        reducible.sort(key=lambda cat: (alloc[cat], raw[cat] - int(raw[cat])), reverse=True)

        index = 0
        while current > sample_records and reducible:
            cat = reducible[index % len(reducible)]
            if alloc[cat] > 1:
                alloc[cat] -= 1
                current -= 1
            index += 1
            reducible = [name for name in reducible if alloc[name] > 1]

    if current < sample_records:
        candidates = sorted(
            categories,
            key=lambda cat: (raw[cat] - alloc[cat], category_sizes[cat] - alloc[cat]),
            reverse=True,
        )
        index = 0
        guard = 0
        while current < sample_records and candidates:
            cat = candidates[index % len(candidates)]
            if alloc[cat] < category_sizes[cat]:
                alloc[cat] += 1
                current += 1
            index += 1
            guard += 1
            if guard > sample_records * max(4, len(candidates)):
                break

        if current < sample_records:
            fallback = [cat for cat in categories if alloc[cat] < category_sizes[cat]]
            for cat in fallback:
                if current >= sample_records:
                    break
                gap = category_sizes[cat] - alloc[cat]
                add = min(gap, sample_records - current)
                alloc[cat] += add
                current += add

    return alloc


def sample_records_from_frame(
    frame: pd.DataFrame,
    sample_records: int | None,
    sample_seed: int,
    sample_strategy: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    total_records = len(frame)
    category_col = "question_category" if "question_category" in frame.columns else None

    if sample_records is None or sample_records >= total_records:
        sampled = frame.copy().reset_index(drop=True)
        return sampled, {
            "strategy": "full",
            "requested_records": sample_records,
            "total_records": total_records,
            "sampled_records": len(sampled),
            "category_distribution_before": _category_distribution(frame, category_col),
            "category_distribution_after": _category_distribution(sampled, category_col),
        }

    if sample_records <= 0:
        raise ValueError("--sample-records must be greater than 0")

    if sample_strategy == "random" or category_col is None:
        sampled = frame.sample(n=sample_records, random_state=sample_seed).reset_index(drop=True)
        return sampled, {
            "strategy": "random",
            "requested_records": sample_records,
            "total_records": total_records,
            "sampled_records": len(sampled),
            "category_distribution_before": _category_distribution(frame, category_col),
            "category_distribution_after": _category_distribution(sampled, category_col),
        }

    category_series = frame[category_col].fillna("unknown").astype(str)
    category_sizes = category_series.value_counts().to_dict()

    if sample_records < len(category_sizes):
        sampled = frame.sample(n=sample_records, random_state=sample_seed).reset_index(drop=True)
        return sampled, {
            "strategy": "random_fallback",
            "fallback_reason": "sample_records is smaller than number of categories",
            "requested_records": sample_records,
            "total_records": total_records,
            "sampled_records": len(sampled),
            "category_distribution_before": _category_distribution(frame, category_col),
            "category_distribution_after": _category_distribution(sampled, category_col),
        }

    alloc = _allocate_stratified_counts(category_sizes, sample_records)
    rng = random.Random(sample_seed)
    sampled_parts: list[pd.DataFrame] = []
    for category, target_n in alloc.items():
        if target_n <= 0:
            continue
        group = frame[category_series == category]
        if target_n >= len(group):
            sampled_parts.append(group)
        else:
            sampled_parts.append(group.sample(n=target_n, random_state=rng.randint(0, 2_147_483_647)))

    sampled = pd.concat(sampled_parts, ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=sample_seed).reset_index(drop=True)

    if len(sampled) > sample_records:
        sampled = sampled.head(sample_records).reset_index(drop=True)

    return sampled, {
        "strategy": "stratified",
        "requested_records": sample_records,
        "total_records": total_records,
        "sampled_records": len(sampled),
        "allocated_per_category": dict(sorted(alloc.items(), key=lambda item: item[0])),
        "category_distribution_before": _category_distribution(frame, category_col),
        "category_distribution_after": _category_distribution(sampled, category_col),
    }


def download_videos_by_ids(
    dataset_id: str,
    revision: str,
    hf_token: str | None,
    local_dir: Path,
    video_ids: list[str],
    hf_endpoint: str | None,
) -> dict[str, Any]:
    downloaded_files: list[str] = []
    missing_video_ids: list[str] = []

    for video_id in video_ids:
        matched = None
        for candidate in build_video_candidates(video_id):
            try:
                hf_hub_download(
                    repo_id=dataset_id,
                    repo_type="dataset",
                    revision=revision,
                    filename=candidate,
                    local_dir=str(local_dir),
                    local_dir_use_symlinks=False,
                    token=hf_token,
                    endpoint=hf_endpoint,
                )
                matched = candidate
                break
            except Exception:
                continue

        if matched is None:
            missing_video_ids.append(video_id)
        else:
            downloaded_files.append(matched)

    return {
        "downloaded_video_files": downloaded_files,
        "missing_video_ids": missing_video_ids,
        "downloaded_video_count": len(downloaded_files),
        "missing_video_count": len(missing_video_ids),
    }


def sample_records_and_download_videos(
    dataset_id: str,
    revision: str,
    hf_token: str | None,
    local_dir: Path,
    sample_records: int | None,
    sample_strategy: str,
    max_videos: int | None,
    sample_seed: int,
    hf_endpoint: str | None,
    keep_full_metadata: bool,
) -> dict[str, Any]:
    parquet_path = resolve_parquet_path(local_dir)
    frame = pd.read_parquet(parquet_path)

    sampled_frame, sampling_info = sample_records_from_frame(
        frame=frame,
        sample_records=sample_records,
        sample_seed=sample_seed,
        sample_strategy=sample_strategy,
    )

    if keep_full_metadata:
        full_metadata_path = local_dir / "metadata_full.parquet"
        frame.to_parquet(full_metadata_path, index=False)

    sampled_frame.to_parquet(parquet_path, index=False)
    sampled_jsonl_path = local_dir / "MCQ_sampled.jsonl"
    sampled_frame.to_json(sampled_jsonl_path, orient="records", lines=True, force_ascii=False)

    video_col = pick_video_column(list(sampled_frame.columns))
    sampled_video_ids = [str(v).strip() for v in sampled_frame[video_col].tolist() if str(v).strip()]
    unique_video_ids = sorted(set(sampled_video_ids))

    if max_videos is not None:
        if max_videos <= 0:
            raise ValueError("--max-videos must be greater than 0")
        random.Random(sample_seed).shuffle(unique_video_ids)
        unique_video_ids = unique_video_ids[:max_videos]

    video_download_info = download_videos_by_ids(
        dataset_id=dataset_id,
        revision=revision,
        hf_token=hf_token,
        local_dir=local_dir,
        video_ids=unique_video_ids,
        hf_endpoint=hf_endpoint,
    )

    sample_manifest = {
        "parquet_path": str(parquet_path),
        "sampled_jsonl_path": str(sampled_jsonl_path),
        "sample_seed": sample_seed,
        "sample_strategy": sample_strategy,
        "record_sampling": sampling_info,
        "unique_video_id_count_in_sample": len(sorted(set(sampled_video_ids))),
        "requested_max_videos": max_videos,
        "selected_video_count": len(unique_video_ids),
        "selected_video_ids": unique_video_ids,
        **video_download_info,
    }

    sample_manifest_path = local_dir / "sampled_videos_manifest.json"
    sample_manifest_path.write_text(json.dumps(sample_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "sample_manifest_path": str(sample_manifest_path),
        "selected_video_count": len(unique_video_ids),
        "downloaded_video_count": video_download_info["downloaded_video_count"],
        "missing_video_count": video_download_info["missing_video_count"],
        "record_sampling": sampling_info,
    }


def main() -> None:
    args = parse_args()
    local_dir = Path(args.local_dir).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    if args.hf_endpoint:
        # 让 huggingface_hub 默认走镜像，便于国内网络稳定下载。
        os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)

    # 仅下载需要的文件，减少无关体积
    patterns = [p.strip() for p in args.allow_patterns.split(",") if p.strip()]

    sample_info: dict[str, object] | None = None
    if args.full_download:
        snapshot_download(
            repo_id=args.dataset_id,
            repo_type="dataset",
            revision=args.revision,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            allow_patterns=patterns if patterns else None,
            token=args.hf_token,
            endpoint=args.hf_endpoint,
        )
    else:
        # 采样模式：先下载元数据，再按记录采样后下载对应视频。
        metadata_patterns = [p for p in patterns if not p.lower().startswith("videos/")]
        if "MCQ.parquet" not in metadata_patterns and "*.parquet" not in metadata_patterns:
            metadata_patterns.append("MCQ.parquet")

        snapshot_download(
            repo_id=args.dataset_id,
            repo_type="dataset",
            revision=args.revision,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            allow_patterns=metadata_patterns if metadata_patterns else None,
            token=args.hf_token,
            endpoint=args.hf_endpoint,
        )

        sample_info = sample_records_and_download_videos(
            dataset_id=args.dataset_id,
            revision=args.revision,
            hf_token=args.hf_token,
            local_dir=local_dir,
            sample_records=args.sample_records,
            sample_strategy=args.sample_strategy,
            max_videos=args.max_videos,
            sample_seed=args.sample_seed,
            hf_endpoint=args.hf_endpoint,
            keep_full_metadata=args.keep_full_metadata,
        )

    # 生成 manifest 便于确认数据完整性
    files = [
        str(path.relative_to(local_dir)).replace("\\", "/")
        for path in local_dir.rglob("*")
        if path.is_file()
    ]
    files.sort()

    manifest = {
        "dataset_id": args.dataset_id,
        "revision": args.revision,
        "local_dir": str(local_dir),
        "hf_endpoint": args.hf_endpoint,
        "allow_patterns": patterns,
        "full_download": args.full_download,
        "sample_records": args.sample_records,
        "sample_strategy": args.sample_strategy,
        "sample_seed": args.sample_seed,
        "file_count": len(files),
        "files": files,
        "sample_mode": sample_info,
    }
    manifest_path = local_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Downloaded dataset snapshot to: {local_dir}")
    print(f"Total files: {len(files)}")
    if sample_info is not None:
        print(
            "Sampled videos: "
            f"selected={sample_info['selected_video_count']}, "
            f"downloaded={sample_info['downloaded_video_count']}, "
            f"missing={sample_info['missing_video_count']}"
        )
        record_sampling = sample_info.get("record_sampling")
        if isinstance(record_sampling, dict):
            print(
                "Sampled records: "
                f"requested={record_sampling.get('requested_records')}, "
                f"sampled={record_sampling.get('sampled_records')}, "
                f"strategy={record_sampling.get('strategy')}"
            )
        print(f"Sample manifest: {sample_info['sample_manifest_path']}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
