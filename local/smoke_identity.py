# Runtime identity probe for the DFlash2 housekeep smoke.
# Runs INSIDE the head container against the installed vLLM tree.
from pathlib import Path

REG = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_dflash2.py")

print("registry module:", REG)
print("exists:", REG.exists())
if not REG.exists():
    raise SystemExit("FAIL: registry module missing")

text = REG.read_text()
checks = {
    "kit marker [glm53-dflash2]": "# [glm53-dflash2]" in text,
    "triton kernel _dflash2_grouped_conv_kernel": "_dflash2_grouped_conv_kernel" in text,
    "op registration dflash2_grouped_conv": "direct_register_custom_op" in text
    and 'op_name="dflash2_grouped_conv"' in text,
    "cuda/eager branch on is_cuda": "hidden_states.is_cuda" in text,
    "torch.topk candidates": "torch.topk" in text,
    "pin lacks get_top_k_tokens": "get_top_k_tokens" not in text,
    "pin lacks draft_logits_spec": "draft_logits_spec" not in text,
    "is_causal override": "is_causal" in text,
}
for name, ok in checks.items():
    print(("PASS " if ok else "FAIL ") + name)
for cls in ("DFlash2Qwen3ForCausalLM", "DFlash2Qwen3Model", "DFlash2DraftModel"):
    print(("PASS class " if ("class " + cls) in text else "FAIL class ") + cls)

from vllm.model_executor.models.registry import _SPECULATIVE_DECODING_MODELS as S

print("registry mapping DFlash2DraftModel:", S.get("DFlash2DraftModel"))

import vllm.model_executor.models.qwen3_dflash2 as m

print("imported module:", m.__file__)
print("DFlash2Qwen3ForCausalLM import:", "ok" if hasattr(m, "DFlash2Qwen3ForCausalLM") else "FAIL")

# The custom op must resolve after importing the module.
import torch

found = []
for ns in ("vllm", "glm53_dflash2"):
    try:
        op = getattr(getattr(torch.ops, ns), "dflash2_grouped_conv", None)
    except Exception:  # noqa: BLE001
        op = None
    if op is not None:
        found.append(ns + "::dflash2_grouped_conv")
print("custom op resolved:", found or "NONE")
print("PASS custom op" if found else "FAIL custom op")

# EAGLE3 aux-hidden interface on the target model.
tgt = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/glm5next.py")
print("glm5next exists:", tgt.exists())
if tgt.exists():
    t = tgt.read_text()
    print("PASS EagleModelMixin" if "EagleModelMixin" in t else "FAIL EagleModelMixin")
    print("PASS aux_hidden_state_layers" if "aux_hidden_state_layers" in t else "FAIL aux_hidden_state_layers")

# DFlash2 speculator module on the pin path.
spec = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py"
)
print("speculator exists:", spec.exists())
if spec.exists():
    s = spec.read_text()
    print("PASS DFlash2Speculator" if "DFlash2Speculator" in s else "FAIL DFlash2Speculator")
print("DONE")
