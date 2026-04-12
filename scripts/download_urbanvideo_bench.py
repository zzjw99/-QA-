#!/usr/bin/env python3
"""Download UrbanVideo-Bench dataset files from Hugging Face Hub."""

# 中文说明：
# - 功能：从 Hugging Face 拉取赛题数据到本地目录。
# - 输入：dataset-id、revision、allow-patterns。
# - 输出：本地数据文件 + download_manifest.json（用于核对下载内容）。

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_dir = Path(args.local_dir).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    # 仅下载需要的文件，减少无关体积
    patterns = [p.strip() for p in args.allow_patterns.split(",") if p.strip()]
    snapshot_download(
        repo_id=args.dataset_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        allow_patterns=patterns if patterns else None,
        token=args.hf_token,
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
    }
    manifest_path = local_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Downloaded dataset snapshot to: {local_dir}")
    print(f"Total files: {len(files)}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
