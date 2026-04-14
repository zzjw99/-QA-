import traceback
import torch
from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
p = r"E:\hf_models\Qwen2.5-VL-3B-Instruct"
off = r"E:\Projects\New project\outputs\offload\acc_test"
try:
    cfg = AutoConfig.from_pretrained(p)
    with init_empty_weights():
        model = Qwen2_5_VLForConditionalGeneration._from_config(cfg)
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=p,
        device_map='auto',
        dtype=torch.float16,
        offload_folder=off,
    )
    print('ACCELERATE_LOAD_OK')
except Exception:
    traceback.print_exc()
