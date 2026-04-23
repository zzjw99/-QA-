#!/usr/bin/env python3
"""Run MCQ inference on Qwen-VL checkpoints and export prediction JSONL."""

# 中文说明：
# - 功能：读取 prepared test/val JSONL，逐条视频问答推理，导出预测。
# - 输入：模型路径 + input-jsonl + data-root。
# - 输出：prediction JSONL（含 prediction 和 prediction_letter）。

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MCQ inference and export prediction JSONL. / 执行 MCQ 推理并导出预测 JSONL。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--backend",
        default="transformers",
        choices=["transformers", "vllm"],
        help="Inference backend / 推理后端",
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
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference requests / 推理批大小",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 means greedy) / 采样温度（0 表示贪心解码）",
    )
    parser.add_argument(
        "--constrained-decoding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable strict MCQ letter decoding constraints / 启用严格 MCQ 字母约束解码",
    )
    parser.add_argument(
        "--append-output-constraint-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append output constraint hint in user prompt / 在用户提示末尾追加输出约束提示",
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
    parser.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM / vLLM 张量并行大小",
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization ratio for vLLM / vLLM 显存利用率",
    )
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=None,
        help="Optional max model length for vLLM / vLLM 最大上下文长度（可选）",
    )
    parser.add_argument(
        "--vllm-max-num-seqs",
        type=int,
        default=32,
        help="Max concurrent sequences in vLLM scheduler / vLLM 调度器最大并发序列数",
    )
    parser.add_argument(
        "--vllm-trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass trust_remote_code to vLLM engine / 向 vLLM 传递 trust_remote_code",
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


def build_user_text(raw_user: str, allowed_letters: list[str], append_hint: bool) -> str:
    user_text = raw_user.replace("<video>", "").strip()
    if not append_hint:
        return user_text

    has_constraint = re.search(
        r"output\s*constraint|reply\s+with\s+only\s+one\s+option\s+letter",
        user_text,
        flags=re.IGNORECASE,
    )
    if has_constraint:
        return user_text

    return (
        f"{user_text}\n\n"
        f"Output constraint: reply with one letter from ({'/'.join(allowed_letters)})."
    )


def get_process_vision_info() -> Callable[[Any], tuple[Any, Any]]:
    try:
        qwen_vl_utils = importlib.import_module("qwen_vl_utils")
        process_vision_info = getattr(qwen_vl_utils, "process_vision_info")
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency qwen-vl-utils. Install with: pip install qwen-vl-utils[decord]"
        ) from exc
    return process_vision_info


def build_video_item(video_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    video_item: dict[str, Any] = {"type": "video", "video": str(video_path)}
    if args.video_fps and args.video_fps > 0:
        video_item["fps"] = args.video_fps
    if args.video_max_pixels is not None:
        video_item["max_pixels"] = args.video_max_pixels
    if args.video_min_pixels is not None:
        video_item["min_pixels"] = args.video_min_pixels
    return video_item


def build_request(sample: dict[str, Any], data_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    conversations = sample.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError(f"Sample {sample.get('id')} has invalid conversations field")

    first_turn = conversations[0]
    if not isinstance(first_turn, dict) or "value" not in first_turn:
        raise ValueError(f"Sample {sample.get('id')} has invalid first conversation turn")

    raw_user = str(first_turn["value"])
    allowed_letters = detect_allowed_letters(sample, raw_user)
    user_text = build_user_text(raw_user, allowed_letters, args.append_output_constraint_prompt)

    video_rel = sample.get("video")
    if not video_rel:
        raise ValueError(f"Sample {sample.get('id')} missing video field")

    video_path = (data_root / str(video_rel)).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    messages = [
        {
            "role": "user",
            "content": [
                build_video_item(video_path, args),
                {"type": "text", "text": user_text},
            ],
        }
    ]

    return {
        "id": sample["id"],
        "allowed_letters": allowed_letters,
        "video_path": str(video_path),
        "messages": messages,
    }


def chunked(items: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def convert_dtype(dtype_str: str) -> torch.dtype | str:
    mapping = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype_str]


def convert_dtype_to_vllm(dtype_str: str) -> str:
    if dtype_str == "auto":
        return "auto"
    return dtype_str


def _resolve_parent_module(model: torch.nn.Module, module_name: str) -> tuple[torch.nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def repair_bnb_layers_with_missing_quant_state(model: torch.nn.Module) -> list[str]:
    try:
        bnb_modules = importlib.import_module("bitsandbytes.nn.modules")
        Linear4bit = getattr(bnb_modules, "Linear4bit")
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


def get_eos_token_ids(model: torch.nn.Module, tokenizer: Any) -> list[int]:
    eos_token_id = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)

    if eos_token_id is None:
        return []
    if isinstance(eos_token_id, int):
        return [int(eos_token_id)]
    return [int(token_id) for token_id in eos_token_id]


def build_single_token_letter_ids(tokenizer: Any, allowed_letters: list[str]) -> list[int]:
    token_ids: set[int] = set()
    for letter in allowed_letters:
        upper = letter.upper()
        variants = [
            upper,
            f" {upper}",
            f"\n{upper}",
            f"({upper})",
            f" ({upper})",
            f"Answer: {upper}",
        ]
        for variant in variants:
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                token_ids.add(int(ids[0]))

    return sorted(token_ids)


def build_prefix_allowed_tokens_fn(
    prompt_lengths: list[int],
    allowed_token_ids_per_sample: list[list[int]],
    eos_token_ids: list[int],
) -> Callable[[int, torch.Tensor], list[int]]:
    def prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.Tensor) -> list[int]:
        prompt_len = prompt_lengths[batch_id]
        generated_len = int(input_ids.shape[0]) - int(prompt_len)
        if generated_len <= 0:
            return allowed_token_ids_per_sample[batch_id]
        if eos_token_ids:
            return eos_token_ids
        return allowed_token_ids_per_sample[batch_id]

    return prefix_allowed_tokens_fn


def build_record(item: dict[str, Any], pred_text: str) -> dict[str, Any]:
    pred_letter = normalize_letter_with_allowed(pred_text, allowed_letters=item["allowed_letters"])
    if pred_letter is None:
        pred_letter = normalize_letter(pred_text)

    return {
        "id": item["id"],
        "prediction": pred_text,
        "prediction_letter": pred_letter,
        "allowed_letters": item["allowed_letters"],
    }


def run_transformers_inference(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    data_root: Path,
    output_path: Path,
) -> None:
    process_vision_info = get_process_vision_info()

    dtype = convert_dtype(args.torch_dtype)
    quantization_config = None
    if args.load_in_4bit:
        try:
            importlib.import_module("bitsandbytes")
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
    model.eval()

    if args.repair_bnb_missing_quant_state:
        repaired_layers = repair_bnb_layers_with_missing_quant_state(model)
        if repaired_layers:
            print(f"Repaired bnb layers without quant_state: {repaired_layers}")

    processor = AutoProcessor.from_pretrained(args.model_name_or_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_batches = math.ceil(len(rows) / args.batch_size)

    with output_path.open("w", encoding="utf-8") as writer, torch.inference_mode():
        for batch_rows in tqdm(chunked(rows, args.batch_size), total=total_batches, desc="Infer(transformers)"):
            items = [build_request(sample, data_root, args) for sample in batch_rows]
            batch_messages = [item["messages"] for item in items]
            batch_texts = [
                processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                for messages in batch_messages
            ]

            image_inputs, video_inputs = process_vision_info(batch_messages)
            model_inputs = processor(
                text=batch_texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            model_inputs = model_inputs.to(model.device)

            prompt_lengths = model_inputs.attention_mask.sum(dim=1).tolist()

            generate_kwargs: dict[str, Any] = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.temperature > 0,
            }
            if args.temperature > 0:
                generate_kwargs["temperature"] = args.temperature

            if args.constrained_decoding:
                allowed_token_ids_per_sample = [
                    build_single_token_letter_ids(processor.tokenizer, item["allowed_letters"])
                    for item in items
                ]
                if all(allowed_token_ids_per_sample):
                    eos_token_ids = get_eos_token_ids(model, processor.tokenizer)
                    generate_kwargs["max_new_tokens"] = 1 if not eos_token_ids else min(args.max_new_tokens, 2)
                    generate_kwargs["min_new_tokens"] = 1
                    generate_kwargs["prefix_allowed_tokens_fn"] = build_prefix_allowed_tokens_fn(
                        prompt_lengths=prompt_lengths,
                        allowed_token_ids_per_sample=allowed_token_ids_per_sample,
                        eos_token_ids=eos_token_ids,
                    )
                else:
                    print(
                        "Warning: constrained decoding fallback triggered in this batch "
                        "(tokenizer could not build single-token letter constraints)."
                    )

            generated_ids = model.generate(**model_inputs, **generate_kwargs)

            generated_trim = [
                generated_ids[index, int(prompt_len) :]
                for index, prompt_len in enumerate(prompt_lengths)
            ]
            pred_texts = processor.batch_decode(
                generated_trim,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            for item, pred_text in zip(items, pred_texts):
                record = build_record(item, pred_text.strip())
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")


def filter_supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs

    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def build_vllm_engine(args: argparse.Namespace) -> Any:
    try:
        LLM = getattr(importlib.import_module("vllm"), "LLM")
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency vllm. Install with: pip install vllm"
        ) from exc

    llm_kwargs: dict[str, Any] = {
        "model": args.model_name_or_path,
        "tensor_parallel_size": args.vllm_tensor_parallel_size,
        "dtype": convert_dtype_to_vllm(args.torch_dtype),
        "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "trust_remote_code": args.vllm_trust_remote_code,
        "max_num_seqs": args.vllm_max_num_seqs,
        "limit_mm_per_prompt": {"video": 1},
    }
    if args.vllm_max_model_len is not None:
        llm_kwargs["max_model_len"] = args.vllm_max_model_len

    filtered_kwargs = filter_supported_kwargs(LLM.__init__, llm_kwargs)
    dropped = sorted(set(llm_kwargs) - set(filtered_kwargs))
    if dropped:
        print(f"Warning: ignoring unsupported vLLM init args: {dropped}")

    return LLM(**filtered_kwargs)


def build_vllm_sampling_params(args: argparse.Namespace, allowed_letters: list[str]) -> Any:
    try:
        SamplingParams = getattr(importlib.import_module("vllm"), "SamplingParams")
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency vllm. Install with: pip install vllm"
        ) from exc

    kwargs: dict[str, Any] = {
        "max_tokens": args.max_new_tokens,
        "temperature": float(args.temperature),
    }
    if args.temperature <= 0:
        kwargs["temperature"] = 0.0

    if args.constrained_decoding and allowed_letters:
        kwargs["guided_choice"] = allowed_letters
        kwargs["max_tokens"] = 1

    try:
        return SamplingParams(**kwargs)
    except TypeError:
        if "guided_choice" in kwargs:
            print("Warning: current vLLM version does not support guided_choice. Fallback to unconstrained decoding.")
            kwargs.pop("guided_choice")
            return SamplingParams(**kwargs)
        raise


def extract_vllm_prediction_text(output: Any) -> str:
    candidate_outputs = getattr(output, "outputs", None)
    if candidate_outputs and len(candidate_outputs) > 0:
        return str(candidate_outputs[0].text).strip()
    return str(output).strip()


def run_vllm_inference(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    data_root: Path,
    output_path: Path,
) -> None:
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    llm = build_vllm_engine(args)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_batches = math.ceil(len(rows) / args.batch_size)

    with output_path.open("w", encoding="utf-8") as writer:
        for batch_rows in tqdm(chunked(rows, args.batch_size), total=total_batches, desc="Infer(vllm)"):
            items = [build_request(sample, data_root, args) for sample in batch_rows]
            for item in items:
                item["prompt"] = processor.apply_chat_template(
                    item["messages"],
                    tokenize=False,
                    add_generation_prompt=True,
                )

            grouped: dict[tuple[str, ...], list[tuple[int, dict[str, Any]]]] = {}
            for index, item in enumerate(items):
                key = tuple(item["allowed_letters"]) if args.constrained_decoding else tuple()
                grouped.setdefault(key, []).append((index, item))

            batch_records: list[dict[str, Any] | None] = [None] * len(items)
            for key, entries in grouped.items():
                allowed_letters = list(key)
                sampling_params = build_vllm_sampling_params(args, allowed_letters)
                prompts = [
                    {
                        "prompt": item["prompt"],
                        "multi_modal_data": {"video": item["video_path"]},
                    }
                    for _, item in entries
                ]

                outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
                for (record_index, item), output in zip(entries, outputs):
                    pred_text = extract_vllm_prediction_text(output)
                    batch_records[record_index] = build_record(item, pred_text)

            for record in batch_records:
                if record is None:
                    raise RuntimeError("Internal error: missing vLLM batch record")
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be >= 1")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be >= 1")
    if args.temperature < 0:
        raise ValueError("--temperature must be >= 0")

    input_path = Path(args.input_jsonl).resolve()
    output_path = Path(args.output_jsonl).resolve()
    data_root = Path(args.data_root).resolve()

    rows = read_jsonl(input_path)
    if not rows:
        raise ValueError(f"No rows found in {input_path}")

    if args.backend == "vllm" and args.load_in_4bit:
        raise ValueError("--load-in-4bit is only supported by --backend transformers")

    if args.backend == "vllm":
        run_vllm_inference(rows=rows, args=args, data_root=data_root, output_path=output_path)
    else:
        run_transformers_inference(rows=rows, args=args, data_root=data_root, output_path=output_path)

    print(f"Prediction file saved to: {output_path}")


if __name__ == "__main__":
    main()
