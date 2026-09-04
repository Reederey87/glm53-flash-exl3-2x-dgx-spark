from __future__ import annotations


class MambaHybridModelState:
    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        # Must reset the speculative acceptance count in this idx which could be stale.
        self.num_accepted_tokens_gpu[req_index].fill_(1)
        if self._align_mode:
            # Seed the running state block from the resumed/prefilled position.
            self._mamba_state_idx_gpu[req_index].fill_(
                (new_req_data.num_computed_tokens - 1) // self.cache_config.block_size
            )

    def _get_mamba_group_info(
        self, kv_cache_config: KVCacheConfig
    ) -> tuple[list[int], MambaSpec]:
        if self._mamba_spec is None:
            group_ids: list[int] = []
            specs: list[MambaSpec] = []
            for i, group in enumerate(kv_cache_config.kv_cache_groups):
                spec = group.kv_cache_spec
                if isinstance(spec, MambaSpec):
                    group_ids.append(i)
                    specs.append(spec)
            assert specs, "no mamba layers in the model"
            assert all(specs[0] == s for s in specs)
            self._mamba_group_ids = group_ids
            self._mamba_spec = specs[0]
        return self._mamba_group_ids, self._mamba_spec

    def preprocess_state(self, kv_cache_config, block_tables):
        mamba_group_ids, mamba_spec = self._get_mamba_group_info(kv_cache_config)
        ctx = self._ensure_align_ctx(kv_cache_config, mamba_group_ids, block_tables)
        return mamba_spec, ctx
