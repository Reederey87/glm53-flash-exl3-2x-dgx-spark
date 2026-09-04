from __future__ import annotations


class Glm5NextIndexer:
    def __init__(self, vllm_config):
        self.index_kpool = 4
        from vllm.v1.attention.backends.mla.indexer import get_max_prefill_buffer_size

        self.max_total_seq_len = get_max_prefill_buffer_size(vllm_config)
