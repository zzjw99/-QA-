#!/usr/bin/env python3
"""Download UrbanVideo-Bench dataset files from Hugging Face Hub."""

# 中文说明：
# - 功能：从 Hugging Face 拉取赛题数据到本地目录。
# - 输入：dataset-id、revision、allow-patterns。
# - 输出：本地数据文件 + download_manifest.json（用于核对下载内容）。

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

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
        "--max-videos",
        type=int,
        default=None,
        help="Optional cap on number of videos to download (downloads metadata + sampled videos) / 可选：限制下载视频数量（会下载元数据 + 采样视频）",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed used when sampling videos / 采样视频时使用的随机种子",
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


def download_sampled_videos(
    dataset_id: str,
    revision: str,
    hf_token: str | None,
    local_dir: Path,
    max_videos: int,
    sample_seed: int,
) -> dict[str, object]:
    parquet_path = local_dir / "MCQ.parquet"
    if not parquet_path.exists():
        candidates = sorted(local_dir.rglob("*.parquet"))
        if not candidates:
            raise FileNotFoundError("No parquet file found. Please ensure metadata is downloaded first.")
        parquet_path = candidates[0]

    frame = pd.read_parquet(parquet_path)
    if "video_id" not in frame.columns:
        raise KeyError(f"Missing video_id column in {parquet_path}")

    video_ids = [str(v).strip() for v in frame["video_id"].tolist() if str(v).strip()]
    unique_video_ids = sorted(set(video_ids))

    random.Random(sample_seed).shuffle(unique_video_ids)
    selected_video_ids = unique_video_ids[:max_videos]

    downloaded_files: list[str] = []
    missing_video_ids: list[str] = []

    for video_id in selected_video_ids:
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
                )
                matched = candidate
                break
            except Exception:
                continue

        if matched is None:
            missing_video_ids.append(video_id)
        else:
            downloaded_files.append(matched)

    sample_manifest = {
        "parquet_path": str(parquet_path),
        "unique_video_id_count": len(unique_video_ids),
        "max_videos": max_videos,
        "sample_seed": sample_seed,
        "selected_video_ids": selected_video_ids,
        "downloaded_video_files": downloaded_files,
        "missing_video_ids": missing_video_ids,
    }
    sample_manifest_path = local_dir / "sampled_videos_manifest.json"
    sample_manifest_path.write_text(json.dumps(sample_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "sample_manifest_path": str(sample_manifest_path),
        "selected_video_count": len(selected_video_ids),
        "downloaded_video_count": len(downloaded_files),
        "missing_video_count": len(missing_video_ids),
    }


def main() -> None:
    args = parse_args()
    local_dir = Path(args.local_dir).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    # 仅下载需要的文件，减少无关体积
    patterns = [p.strip() for p in args.allow_patterns.split(",") if p.strip()]

    sample_info: dict[str, object] | None = None
    if args.max_videos is None:
        snapshot_download(
            repo_id=args.dataset_id,
            repo_type="dataset",
            revision=args.revision,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            allow_patterns=patterns if patterns else None,
            token=args.hf_token,
        )
    else:
        if args.max_videos <= 0:
            raise ValueError("--max-videos must be greater than 0")

        metadata_patterns = [p for p in patterns if not p.lower().startswith("videos/")]
        if "MCQ.parquet" not in metadata_patterns:
            metadata_patterns.append("MCQ.parquet")

        snapshot_download(
            repo_id=args.dataset_id,
            repo_type="dataset",
            revision=args.revision,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            allow_patterns=metadata_patterns if metadata_patterns else None,
            token=args.hf_token,
        )

        sample_info = download_sampled_videos(
            dataset_id=args.dataset_id,
            revision=args.revision,
            hf_token=args.hf_token,
            local_dir=local_dir,
            max_videos=args.max_videos,
            sample_seed=args.sample_seed,
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
        "allow_patterns": patterns,
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
        print(f"Sample manifest: {sample_info['sample_manifest_path']}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
