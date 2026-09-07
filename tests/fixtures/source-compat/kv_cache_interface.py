"""Compact live KVCacheSpec.merge + MLA contracts."""
from __future__ import annotations

import copy


class KVCacheSpec:
    def __init__(self, block_size: int, extra: str = "same") -> None:
        self.block_size = block_size
        self.extra = extra

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, KVCacheSpec)
            and self.block_size == other.block_size
            and self.extra == other.extra
        )

    @classmethod
    def merge(cls, specs: list[KVCacheSpec]) -> KVCacheSpec:
        """
        Merge a list of KVCacheSpec objects into a single KVCacheSpec object.
        """
        assert all(spec == specs[0] for spec in specs[1:]), (
            "All layers in the same KV cache group must be the same."
        )
        return copy.deepcopy(specs[0])


class MLAAttentionSpec(KVCacheSpec):
    def __init__(
        self,
        block_size: int,
        extra: str = "same",
        cache_dtype_str: str | None = None,
        non_causal_multi_token_decode: bool = False,
    ) -> None:
        super().__init__(block_size, extra)
        self.cache_dtype_str = cache_dtype_str
        self.non_causal_multi_token_decode = non_causal_multi_token_decode

    @classmethod
    def merge(cls, specs: list[MLAAttentionSpec]) -> MLAAttentionSpec:
        return cls(
            block_size=specs[0].block_size,
            non_causal_multi_token_decode=any(
                spec.non_causal_multi_token_decode for spec in specs
            ),
        )

    @property
    def real_page_size_bytes(self) -> int:
        if self.cache_dtype_str == "fp8_ds_mla":
            # V3.2 main MLA: 656-byte custom layout (kv_lora_rank=512 +
            # qk_rope_head_dim=64, head_size=576). See flashmla_sparse.py.
            return self.block_size * 656
        return self.block_size * 584
