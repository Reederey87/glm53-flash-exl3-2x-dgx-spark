#!/usr/bin/env python3
"""Teach Glm5Next the EAGLE3 aux-hidden interface DFlash2 uses.

Fail-closed, idempotent, env-overridable. The image self-check pins the
installed strings (``EagleModelMixin``, ``SupportsEagle3``, ``hc_contract``,
``aux_hidden_state_layers``). Anchors are the glm53-flash pin's
``models/glm5next/nvidia/model.py``.
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path


SITE = Path(
    os.environ.get(
        "GLM53_VLLM_SITE",
        "/usr/local/lib/python3.12/dist-packages/vllm",
    )
)
TARGET = Path(
    os.environ.get(
        "GLM53_GLM5NEXT_MODEL_PY",
        str(SITE / "models/glm5next/nvidia/model.py"),
    )
)

EDITS: tuple[tuple[str, str], ...] = (
    (
        "from vllm.model_executor.models.interfaces import (\n"
        "    HasInnerState,\n"
        "    IsHybrid,\n"
        "    MixtureOfExperts,\n"
        "    SupportsPP,\n"
        ")\n",
        "from vllm.model_executor.models.interfaces import (\n"
        "    EagleModelMixin,\n"
        "    HasInnerState,\n"
        "    IsHybrid,\n"
        "    MixtureOfExperts,\n"
        "    SupportsEagle3,\n"
        "    SupportsPP,\n"
        ")\n",
    ),
    (
        "class Glm5NextModel(nn.Module):\n",
        "class Glm5NextModel(nn.Module, EagleModelMixin):\n",
    ),
    (
        "        self._active_layers = self.layers[self.start_layer : self.end_layer]\n",
        "        self._active_layers = self.layers[self.start_layer : self.end_layer]\n"
        "        self.aux_hidden_state_layers: tuple[int, ...] = ()\n",
    ),
    (
        "        full_num_tokens = positions.shape[0]\n"
        "        if self.is_sequence_parallel:\n"
        "            hidden_states = sp_shard(hidden_states)\n"
        "\n"
        "        for layer in self._active_layers:\n"
        "            hidden_states, residual, post, comb = layer(\n"
        "                positions, hidden_states, residual, post, comb\n"
        "            )\n",
        "        full_num_tokens = positions.shape[0]\n"
        "        if self.is_sequence_parallel:\n"
        "            hidden_states = sp_shard(hidden_states)\n"
        "\n"
        "        aux_hidden_states: list[torch.Tensor] = []\n"
        "        for idx, layer in enumerate(\n"
        "            self._active_layers, start=self.start_layer\n"
        "        ):\n"
        "            hidden_states, residual, post, comb = layer(\n"
        "                positions, hidden_states, residual, post, comb\n"
        "            )\n"
        "            if idx + 1 not in self.aux_hidden_state_layers:\n"
        "                continue\n"
        "            # Mid-stack mHC defers hc_post; materialize then contract\n"
        "            # 4 streams -> [tokens, hidden] (deepseek_v4 eagle3 pattern).\n"
        "            if post is not None and hasattr(layer, \"hc_post\"):\n"
        "                value = hc_contract(\n"
        "                    layer.hc_post(hidden_states, residual, post, comb),\n"
        "                    layer.n,\n"
        "                )\n"
        "            else:\n"
        "                value = hidden_states\n"
        "                if value.ndim == 3:\n"
        "                    value = value.mean(dim=1)\n"
        "            if self.is_sequence_parallel:\n"
        "                value = sp_all_gather(value)[:full_num_tokens]\n"
        "            aux_hidden_states.append(value)\n",
    ),
    (
        "        hidden_states = self.norm(hidden_states)\n"
        "        return hidden_states\n",
        "        hidden_states = self.norm(hidden_states)\n"
        "        if aux_hidden_states:\n"
        "            return hidden_states, aux_hidden_states\n"
        "        return hidden_states\n",
    ),
    (
        "class Glm5NextForCausalLM(\n"
        "    nn.Module, HasInnerState, SupportsPP, MixtureOfExperts, IsHybrid\n"
        "):\n",
        "class Glm5NextForCausalLM(\n"
        "    nn.Module,\n"
        "    HasInnerState,\n"
        "    SupportsPP,\n"
        "    MixtureOfExperts,\n"
        "    IsHybrid,\n"
        "    SupportsEagle3,\n"
        "):\n",
    ),
    (
        "class Glm5NextForConditionalGeneration(\n"
        "    Glm4vForConditionalGeneration, HasInnerState, IsHybrid\n"
        "):\n",
        "class Glm5NextForConditionalGeneration(\n"
        "    Glm4vForConditionalGeneration, HasInnerState, IsHybrid, SupportsEagle3\n"
        "):\n",
    ),
)


def counts(text: str) -> tuple[list[int], list[int]]:
    old = [text.count(old) for old, _ in EDITS]
    new = [text.count(new) for _, new in EDITS]
    return old, new


def leftover_old_counts() -> list[int]:
    """Old anchors that remain as a prefix/substring of their replacement."""
    return [new.count(old) for old, new in EDITS]


def verified_state(text: str) -> bool:
    old, new = counts(text)
    return (
        old == leftover_old_counts()
        and new == [1] * len(EDITS)
        and "class Glm5NextModel(nn.Module, EagleModelMixin):" in text
        and "SupportsEagle3" in text
        and "aux_hidden_state_layers" in text
        and "layer.hc_post(hidden_states, residual, post, comb)" in text
        and "hc_contract(" in text
        and "return hidden_states, aux_hidden_states" in text
    )


def prepare(source: str) -> tuple[str, str]:
    old, new = counts(source)
    if verified_state(source):
        return source, "already present"
    if old != [1] * len(EDITS) or any(n != 0 for n in new):
        raise ValueError(
            "pinned glm5next EAGLE3 anchors drifted "
            f"(old={old}, new={new})"
        )
    patched = source
    for old_s, new_s in EDITS:
        patched = patched.replace(old_s, new_s, 1)
    if not verified_state(patched):
        raise ValueError("glm5next EAGLE3 post-patch verification failed")
    return patched, "patched"


def replace_file(target: Path, source: str) -> None:
    tmp = target.with_name(f".{target.name}.glm53-eagle3.tmp")
    try:
        tmp.write_text(source)
        os.chmod(tmp, stat.S_IMODE(target.stat().st_mode))
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def main() -> int:
    if not TARGET.is_file():
        raise SystemExit(f"missing {TARGET}")
    source = TARGET.read_text()
    try:
        patched, action = prepare(source)
    except ValueError as exc:
        raise SystemExit(f"glm5next EAGLE3 preflight failed: {exc}") from exc
    compile(patched, str(TARGET), "exec")
    if patched != source:
        replace_file(TARGET, patched)
    print(f"{TARGET.name}: glm5next EAGLE3 aux-hidden {action}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
