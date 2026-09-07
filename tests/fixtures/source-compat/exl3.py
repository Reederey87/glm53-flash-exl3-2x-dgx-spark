"""Compact live EXL3 paths: fused MoE is C4 decode; index_select is overflow."""


def apply_exl3_fused_moe(x2d):
    return x2d


def apply_exl3_batched_fat(xh, token_idx, h):
    import torch

    torch.index_select(xh, 0, token_idx, out=h)
    return h
