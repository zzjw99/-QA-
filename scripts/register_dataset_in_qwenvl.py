#!/usr/bin/env python3
"""Register custom JSONL datasets into QwenVL finetune data registry."""

# 中文说明：
# - 功能：把本地 JSONL 数据注册到 Qwen 框架 data_dict 中。
# - 实现：在 __init__.py 中插入/更新 AUTO_REGISTERED_DATASETS 标记块。
# - 优点：重复运行时会覆盖标记块，不会无限追加脏内容。

from __future__ import annotations

import argparse
from pathlib import Path


START_MARK = "# >>> AUTO_REGISTERED_DATASETS >>>"
END_MARK = "# <<< AUTO_REGISTERED_DATASETS <<<"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register local JSONL datasets into QwenVL data registry. / 将本地 JSONL 数据集注册到 QwenVL 数据注册表。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--qwenvl-data-init",
        default="third_party/Qwen2.5-VL/qwen-vl-finetune/qwenvl/data/__init__.py",
        help="Path to qwenvl/data/__init__.py / qwenvl/data/__init__.py 文件路径",
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Base data path used by QwenVL data loader / QwenVL 数据加载器使用的数据根路径",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Dataset mapping alias=annotation_jsonl_path, can be repeated / 数据映射，格式 alias=annotation_jsonl_path，可重复传入",
    )
    return parser.parse_args()


def parse_dataset_mapping(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Invalid --dataset format: {value}. Expected alias=path")
    alias, path = value.split("=", 1)
    alias = alias.strip()
    if not alias:
        raise ValueError(f"Invalid alias in --dataset: {value}")
    annotation_path = Path(path).expanduser().resolve()
    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
    return alias, annotation_path


def render_block(mappings: list[tuple[str, Path]], data_path: Path) -> str:
    lines = [START_MARK, "AUTO_REGISTERED_DATASETS = {"]
    for alias, ann in mappings:
        lines.extend(
            [
                f'    "{alias}": {{',
                f'        "annotation_path": r"{ann.as_posix()}",',
                f'        "data_path": r"{data_path.as_posix()}",',
                "    },",
            ]
        )
    lines.extend(["}", "data_dict.update(AUTO_REGISTERED_DATASETS)", END_MARK, ""])
    return "\n".join(lines)


def inject_block(content: str, block: str) -> str:
    start = content.find(START_MARK)
    end = content.find(END_MARK)
    # 如果已有自动注册块，则替换；否则追加到文件末尾
    if start != -1 and end != -1 and end > start:
        end = end + len(END_MARK)
        head = content[:start].rstrip() + "\n\n"
        tail = content[end:].lstrip("\n")
        return head + block + tail
    return content.rstrip() + "\n\n" + block


def main() -> None:
    args = parse_args()
    init_path = Path(args.qwenvl_data_init).resolve()
    if not init_path.exists():
        raise FileNotFoundError(f"QwenVL data registry file not found: {init_path}")

    data_path = Path(args.data_path).expanduser().resolve()
    mappings = [parse_dataset_mapping(item) for item in args.dataset]

    original = init_path.read_text(encoding="utf-8")
    block = render_block(mappings, data_path)
    updated = inject_block(original, block)
    init_path.write_text(updated, encoding="utf-8")

    print(f"Updated dataset registry: {init_path}")
    for alias, ann in mappings:
        print(f"  - {alias} -> {ann}")
    print(f"Data path: {data_path}")


if __name__ == "__main__":
    main()
