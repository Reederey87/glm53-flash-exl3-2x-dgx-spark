#!/usr/bin/env python3
"""Teach Glm5Next the EAGLE3 aux-hidden interface DFlash2 uses."""

from __future__ import annotations

from pathlib import Path

SITE = Path("/usr/local/lib/python3.12/dist-packages/vllm")
TARGET = SITE / "models/glm5next/nvidia/model.py"


def replace_once(old: str, new: str) -> None:
    text = TARGET.read_text()
    if new in text and old not in text:
        return
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{TARGET}: expected one patch target, found {n}: {old!r}")
    TARGET.write_text(text.replace(old, new))


def main() -> None:
    replace_once(
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
    )
    replace_once(
        "class Glm5NextModel(nn.Module):\n",
        "class Glm5NextModel(nn.Module, EagleModelMixin):\n",
    )
    replace_once(
        "        self._active_layers = self.layers[self.start_layer : self.end_layer]\n",
        "        self._active_layers = self.layers[self.start_layer : self.end_layer]\n"
        "        self.aux_hidden_state_layers: tuple[int, ...] = ()\n",
    )
    replace_once(
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
        "            aux_hidden_states.append(value)\n"
    )
    replace_once(
        "        hidden_states = self.norm(hidden_states)\n"
        "        return hidden_states\n",
        "        hidden_states = self.norm(hidden_states)\n"
        "        if aux_hidden_states:\n"
        "            return hidden_states, aux_hidden_states\n"
        "        return hidden_states\n",
    )
    replace_once(
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
    )
    replace_once(
        "class Glm5NextForConditionalGeneration(\n"
        "    Glm4vForConditionalGeneration, HasInnerState, IsHybrid\n"
        "):\n",
        "class Glm5NextForConditionalGeneration(\n"
        "    Glm4vForConditionalGeneration, HasInnerState, IsHybrid, SupportsEagle3\n"
        "):\n",
    )
    compile(TARGET.read_text(), str(TARGET), "exec")
    print("glm5next EAGLE3 aux-hidden overlay installed")


if __name__ == "__main__":
    main()
