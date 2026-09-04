from __future__ import annotations


class GPUModelRunner:
    def initialize_kv_cache(self):
        self.attn_groups, attn_cg_support, self.kernel_block_sizes = init_attn_backend(
            self.kv_cache_config, self.vllm_config, self.device
        )
        attn_cg_support = attn_cg_support.narrow(
            *self.model_state.get_additional_cg_support()
        )
        return attn_cg_support
