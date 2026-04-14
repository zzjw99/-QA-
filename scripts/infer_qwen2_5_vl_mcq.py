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
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

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
    parser.add_argument(
        "--low-cpu-mem-usage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable low CPU memory loading / 启用低 CPU 内存加载",
    )
    parser.add_argument(
        "--offload-state-dict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Offload state dict to disk during load / 加载时将 state dict 临时卸载到磁盘",
    )
    parser.add_argument(
        "--offload-folder",
        default="outputs/offload",
        help="Disk folder used for offload buffers / 用于离线缓存的磁盘目录",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=1.0,
        help="Video sampling fps for vision encoder / 视频采样帧率（降低可减小显存）",
    )
    parser.add_argument(
        "--video-max-pixels",
        type=int,
        default=50176,
        help="Max pixels per frame passed to processor / 每帧最大像素数（降低可减小显存）",
    )
    parser.add_argument(
        "--video-min-pixels",
        type=int,
        default=None,
        help="Optional min pixels per frame for processor / 每帧最小像素数（可选）",
    )
    parser.add_argument(
        "--repair-bnb-missing-quant-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Repair broken bnb 4bit layers with missing quant_state / 修复缺失 quant_state 的 bnb 4bit 层",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model with 4-bit quantization (bitsandbytes) / 使用 bitsandbytes 4bit 量化加载模型",
    )
    parser.add_argument(
        "--bnb-4bit-quant-type",
        default="nf4",
        choices=["nf4", "fp4"],
        help="4-bit quantization type / 4bit 量化类型",
    )
    parser.add_argument(
        "--bnb-4bit-compute-dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Compute dtype for 4-bit kernels / 4bit 计算核的 dtype",
    )
    parser.add_argument(
        "--bnb-4bit-use-double-quant",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use double quantization in bitsandbytes / bitsandbytes 双重量化开关",
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
    return normalize_letter_with_allowed(text, allowed_letters=None)


def normalize_letter_with_allowed(text: str, allowed_letters: list[str] | None) -> str | None:
    value = text.strip().upper()
    allowed = {letter.upper() for letter in allowed_letters} if allowed_letters else None
    patterns = [r"\(([A-Z])\)", r"\b([A-Z])\b", r"OPTION\s*([A-Z])", r"ANSWER\s*[:：]?\s*([A-Z])"]
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            candidate = match.group(1).upper()
            if allowed and candidate not in allowed:
                continue
            return candidate
    return None


def extract_option_letters(prompt_text: str) -> list[str]:
    marker = re.search(r"(CHOICE|CHOOSE)\s*[:：]", prompt_text, flags=re.IGNORECASE)
    segment = prompt_text[marker.end() :] if marker else prompt_text
    letters = re.findall(r"(?:^|\n)\s*([A-Z])\.\s", segment, flags=re.IGNORECASE)
    if not letters:
        letters = re.findall(r"\b([A-Z])\.\s", segment, flags=re.IGNORECASE)

    ordered: list[str] = []
    for letter in letters:
        value = letter.upper()
        if value not in ordered:
            ordered.append(value)
    return ordered


def detect_allowed_letters(sample: dict[str, Any], raw_user: str) -> list[str]:
    value = sample.get("allowed_letters")
    if isinstance(value, list):
        from_sample = [str(item).strip().upper() for item in value if str(item).strip()]
        if from_sample:
            return from_sample

    from_prompt = extract_option_letters(raw_user)
    if from_prompt:
        return from_prompt

    return ["A", "B", "C", "D"]


def convert_dtype(dtype_str: str) -> torch.dtype | str:
    mapping = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype_str]


def _resolve_parent_module(model: torch.nn.Module, module_name: str) -> tuple[torch.nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def repair_bnb_layers_with_missing_quant_state(model: torch.nn.Module) -> list[str]:
    try:
        from bitsandbytes.nn.modules import Linear4bit
    except ImportError:
        return []

    repaired: list[str] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, Linear4bit):
            continue
        if getattr(module, "quant_state", None) is not None:
            continue
        if not isinstance(module.weight, torch.nn.Parameter):
            continue

        parent, child_name = _resolve_parent_module(model, name)
        replacement = torch.nn.Linear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        replacement.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            replacement.bias.data.copy_(module.bias.data)
        setattr(parent, child_name, replacement)
        repaired.append(name)

    return repaired


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl).resolve()
    output_path = Path(args.output_jsonl).resolve()
    data_root = Path(args.data_root).resolve()

    rows = read_jsonl(input_path)
    if not rows:
        raise ValueError(f"No rows found in {input_path}")

    dtype = convert_dtype(args.torch_dtype)

    quantization_config = None
    if args.load_in_4bit:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "Missing dependency bitsandbytes. Install with: pip install bitsandbytes"
            ) from exc

        compute_dtype = convert_dtype(args.bnb_4bit_compute_dtype)
        if isinstance(compute_dtype, str):
            compute_dtype = torch.float16

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        )
        if dtype == "auto":
            dtype = torch.float16

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map=args.device_map,
        quantization_config=quantization_config,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        offload_state_dict=args.offload_state_dict,
        offload_folder=str(Path(args.offload_folder).resolve()) if args.offload_folder else None,
    )

    if args.repair_bnb_missing_quant_state:
        repaired_layers = repair_bnb_layers_with_missing_quant_state(model)
        if repaired_layers:
            print(f"Repaired bnb layers without quant_state: {repaired_layers}")

    processor = AutoProcessor.from_pretrained(args.model_name_or_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as writer:
        for sample in tqdm(rows, desc="Infer"):
            raw_user = sample["conversations"][0]["value"]
            allowed_letters = detect_allowed_letters(sample, raw_user)
            user_text = raw_user.replace("<video>", "").strip()
            user_text = f"{user_text}\n\nOutput constraint: reply with one letter from ({'/'.join(allowed_letters)})."

            video_rel = sample["video"]
            video_path = (data_root / video_rel).resolve()
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")

            video_item: dict[str, Any] = {"type": "video", "video": str(video_path)}
            if args.video_fps and args.video_fps > 0:
                video_item["fps"] = args.video_fps
            if args.video_max_pixels is not None:
                video_item["max_pixels"] = args.video_max_pixels
            if args.video_min_pixels is not None:
                video_item["min_pixels"] = args.video_min_pixels

            # 构造 Qwen-VL chat 输入：视频 + 文本问题
            messages = [
                {
                    "role": "user",
                    "content": [
                        video_item,
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
            pred_letter = normalize_letter_with_allowed(pred_text, allowed_letters=allowed_letters)
            if pred_letter is None:
                pred_letter = normalize_letter(pred_text)

            # 统一输出格式，便于后续评测脚本对齐 id
            record = {
                "id": sample["id"],
                "prediction": pred_text,
                "prediction_letter": pred_letter,
                "allowed_letters": allowed_letters,
            }
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"Prediction file saved to: {output_path}")


if __name__ == "__main__":
    main()
