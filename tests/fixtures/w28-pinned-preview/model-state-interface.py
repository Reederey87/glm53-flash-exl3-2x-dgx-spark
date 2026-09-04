from __future__ import annotations


class ModelState:
    def apply_staged_writes(self) -> None:
        return None

    def get_additional_cg_support(self) -> tuple[AttentionCGSupport, str | None]:
        return AttentionCGSupport.ALWAYS, None
