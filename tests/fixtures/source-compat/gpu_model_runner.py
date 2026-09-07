"""Compact V2 GPU runner after the W28 overlay."""


class GPUModelRunner:
    def initialize_kv_cache(self) -> None:
        self.model_state.set_kv_cache_config(  # [glm53-w28-correctness]
            self.kv_cache_config
        )
