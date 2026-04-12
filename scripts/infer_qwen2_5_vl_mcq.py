#!/usr/bin/env python3
"""Run MCQ inference on Qwen2.5-VL checkpoints and export prediction JSONL."""

# 中文说明：
# - 功能：读取 prepared test/val JSONL，逐条视频问答推理，导出预测。
# - 输入：模型路径 + input-jsonl + data-root。
# - 输出：prediction JSONL（含 prediction 和 prediction_letter）。

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

try:
    from qwen_vl_utils import process_vision_info
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency qwen-vl-utils. Install with: pip install qwen-vl-utils[decord]"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MCQ inference and export prediction JSONL. / 执行 MCQ 推理并导出预测 JSONL。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-name-or-path",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Model name or local checkpoint path / 模型名称或本地权重路径",
    )
    parser.add_argument(
        "--input-jsonl",
        required=True,
        help="Prepared dataset split JSONL / 预处理后的数据切分 JSONL",
    )
    parser.add_argument(
        "--output-jsonl",
        required=True,
        help="Output prediction JSONL path / 预测结果 JSONL 输出路径",
    )
    parser.add_argument(
        "--data-root",
        default="data/raw/urbanvideo_bench",
        help="Base path used to resolve relative video paths / 用于解析相对视频路径的数据根目录",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Max generated tokens per sample / 每个样本最大生成 token 数",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 means greedy) / 采样温度（0 表示贪心解码）",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map setting / Transformers 的 device_map 参数",
    )
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype for model loading / 模型加载时使用的精度类型",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_letter(text: str) -> str | None:
    value = text.strip().upper()
    patterns = [r"\(([A-D])\)", r"\b([A-D])\b", r"OPTION\s*([A-D])", r"ANSWER\s*[:：]?\s*([A-D])"]
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            return match.group(1)
    return None


def convert_dtype(dtype_str: str) -> torch.dtype | str:
    mapping = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype_str]


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl).resolve()
    output_path = Path(args.output_jsonl).resolve()
    data_root = Path(args.data_root).resolve()

    rows = read_jsonl(input_path)
    if not rows:
        raise ValueError(f"No rows found in {input_path}")

    dtype = convert_dtype(args.torch_dtype)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map=args.device_map,
    )
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as writer:
        for sample in tqdm(rows, desc="Infer"):
            raw_user = sample["conversations"][0]["value"]
            user_text = raw_user.replace("<video>", "").strip()

            video_rel = sample["video"]
            video_path = (data_root / video_rel).resolve()
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")

            # 构造 Qwen-VL chat 输入：视频 + 文本问题
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": str(video_path)},
                        {"type": "text", "text": user_text},
                    ],
                }
            ]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)

            model_inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            model_inputs = model_inputs.to(model.device)

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
            )

            generated_trim = generated_ids[:, model_inputs.input_ids.shape[1] :]
            pred_text = processor.batch_decode(
                generated_trim,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            pred_letter = normalize_letter(pred_text)

            # 统一输出格式，便于后续评测脚本对齐 id
            record = {
                "id": sample["id"],
                "prediction": pred_text,
                "prediction_letter": pred_letter,
            }
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Prediction file saved to: {output_path}")


if __name__ == "__main__":
    main()
