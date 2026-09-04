from __future__ import annotations


class Buffer:
    def __init__(self, size):
        self.size = size

    def __getitem__(self, item):
        return range(min(item.stop, self.size))


def gather(prefill_metadata, short_prefill, total_seq_lens, index_kpool):
    k_quant_full = Buffer(total_seq_lens)
    k_scale_full = Buffer(total_seq_lens)
    if True:
        for chunk in prefill_metadata.chunks if not short_prefill else ():
            k_quant = k_quant_full[: chunk.total_seq_lens]
            k_scale = k_scale_full[: chunk.total_seq_lens]
        return len(k_quant), len(k_scale)
