# Runtime identity probe for the DFlash2 housekeep smoke.
# Runs INSIDE the head container against the installed vLLM tree.
#
# Paths are the pin 487ecf187 layout, which is NOT uniform:
#   * the drafter model + its parent live under vllm/model_executor/models/
#   * the glm5next target model lives under vllm/models/glm5next/nvidia/
# Checking the wrong parent silently skips a target, so each path is asserted
# to exist before it is inspected.
import re
import sys
from pathlib import Path

SITE = Path("/usr/local/lib/python3.12/dist-packages/vllm")
REG = SITE / "model_executor/models/qwen3_dflash2.py"
PARENT = SITE / "model_executor/models/qwen3_dflash.py"
GLM5NEXT = SITE / "models/glm5next/nvidia/model.py"
SPECULATOR = SITE / "v1/worker/gpu/spec_decode/dflash2/speculator.py"

FAILURES: list[str] = []


def check(name: str, ok: bool) -> None:
    if not ok:
        FAILURES.append(name)
    print(("PASS " if ok else "FAIL ") + name)


# --- installed drafter module -------------------------------------------------
print("drafter module:", REG)
check("drafter module exists", REG.is_file())
text = REG.read_text() if REG.is_file() else ""

check("kit marker [glm53-dflash2]", "# [glm53-dflash2]" in text)
check("triton kernel _dflash2_grouped_conv_kernel", "_dflash2_grouped_conv_kernel" in text)
check(
    "op registration dflash2_grouped_conv",
    "direct_register_custom_op" in text and 'op_name="dflash2_grouped_conv"' in text,
)
check("CUDA/eager branch on is_cuda", "hidden_states.is_cuda" in text)
check("torch.topk candidate selection", "torch.topk" in text)

# Absence checks must look for EXECUTABLE use. A bare substring match trips on
# the module docstring, which names both pin APIs it deliberately does not call.
check("no executable .get_top_k_tokens( call", not re.search(r"\.get_top_k_tokens\s*\(", text))
check(
    "no draft_logits_spec definition or call",
    not re.search(r"def\s+draft_logits_spec|\.draft_logits_spec\s*\(", text),
)

for cls in ("DFlash2Qwen3ForCausalLM", "DFlash2Qwen3Model", "DFlash2Qwen3DecoderLayer"):
    check(f"class {cls}", f"class {cls}" in text)

# --- registry contract --------------------------------------------------------
# DFlash2DraftModel is an ARCHITECTURE key, not a class name: the mapping is the
# contract, and it must point at the pinned installed path.
from vllm.model_executor.models.registry import _SPECULATIVE_DECODING_MODELS as SPECS

mapping = SPECS.get("DFlash2DraftModel")
print("registry mapping DFlash2DraftModel:", mapping)
check(
    "registry maps to the pinned installed path",
    mapping == ("qwen3_dflash2", "DFlash2Qwen3ForCausalLM"),
)

import vllm.model_executor.models.qwen3_dflash2 as module

print("imported module:", module.__file__)
check("module imports", module.__file__ == str(REG))
check("DFlash2Qwen3ForCausalLM importable", hasattr(module, "DFlash2Qwen3ForCausalLM"))

# --- custom op resolves -------------------------------------------------------
import torch

resolved = [
    ns
    for ns in ("vllm", "glm53_dflash2")
    if getattr(getattr(torch.ops, ns, None), "dflash2_grouped_conv", None) is not None
]
print("custom op resolved:", [f"{ns}::dflash2_grouped_conv" for ns in resolved] or "NONE")
check("custom op resolves after import", bool(resolved))

# --- parent module: the is_causal / decoder_layer_cls overrides live here ------
print("parent module:", PARENT)
check("parent qwen3_dflash.py exists", PARENT.is_file())
parent = PARENT.read_text() if PARENT.is_file() else ""
check("parent honors checkpoint is_causal", "is_causal" in parent)
check("parent sets decoder_layer_cls", "decoder_layer_cls" in parent)

# --- EAGLE3 aux-hidden on the target model -----------------------------------
# This is the target model's aux-hidden interface that DFlash2 consumes, not a
# second speculator.
print("glm5next target model:", GLM5NEXT)
check("glm5next target model exists", GLM5NEXT.is_file())
glm = GLM5NEXT.read_text() if GLM5NEXT.is_file() else ""
check("EagleModelMixin", "EagleModelMixin" in glm)
check("SupportsEagle3", "SupportsEagle3" in glm)
check("aux_hidden_state_layers", "aux_hidden_state_layers" in glm)

# --- speculator ---------------------------------------------------------------
print("speculator:", SPECULATOR)
check("DFlash2Speculator exists", SPECULATOR.is_file() and "DFlash2Speculator" in SPECULATOR.read_text())

# --- verdict ------------------------------------------------------------------
if FAILURES:
    print(f"IDENTITY FAIL — {len(FAILURES)} check(s): {FAILURES}")
    sys.exit(1)
print("IDENTITY PASS")
