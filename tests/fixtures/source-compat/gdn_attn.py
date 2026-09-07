"""Compact live GDN speculative-row classification."""


class GDNAttention:
    def build_decode_metadata(self, num_decode_draft_tokens_cpu) -> None:
        # Live GDN classifies speculative rows by draft-count tags, not
        # BaseMamba padded-tail bookkeeping (#55178 is parked).
        self.num_decode_draft_tokens_cpu = num_decode_draft_tokens_cpu
