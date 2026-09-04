from __future__ import annotations


class FlashInferMLASparseSM120Impl:
    """SM120 FlashInfer sparse-MLA implementation."""

    is_sparse = True
    supports_dense_mha_prefill = False

    def __init__(
        self,
        num_heads: int,
    ) -> None:
        self.num_heads = num_heads
