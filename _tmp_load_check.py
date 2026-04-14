import traceback
from safetensors.torch import load_file
p = r"E:\hf_models\Qwen2.5-VL-3B-Instruct\model-00002-of-00002.safetensors"
try:
    d = load_file(p)
    print('OK', len(d))
except Exception:
    traceback.print_exc()
