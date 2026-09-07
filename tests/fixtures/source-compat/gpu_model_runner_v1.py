"""Compact V1 GPU runner. W28 must not land here."""


class GPUModelRunner:
    def initialize_kv_cache(self) -> None:
        self.attn_groups = []
