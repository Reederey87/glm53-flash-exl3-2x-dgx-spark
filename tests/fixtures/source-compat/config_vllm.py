"""Compact live V2-runner architecture default."""

DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES = frozenset(
    {
        "DeepseekV2ForCausalLM",
        "DeepseekV4ForCausalLM",
        "Glm5NextForCausalLM",
        "Glm5NextForConditionalGeneration",
    }
)
