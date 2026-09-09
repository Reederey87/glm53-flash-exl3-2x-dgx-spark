# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501


import torch
import torch.nn as nn

from vllm.model_executor.custom_op import CustomOp
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import RCP_LN2, cdiv, next_power_of_2

from .chunk_delta_h import chunk_gated_delta_rule_fwd_h
from .cumsum import chunk_local_cumsum
from .fused_recurrent import fused_recurrent_gated_delta_rule_fwd_kernel
from .index import prepare_chunk_indices
from .l2norm import l2norm_fwd
from .op import exp2, log
from .solve_tril import solve_tril
from .utils import FLA_CHUNK_SIZE, is_amd

BT_LIST_AUTOTUNE = [32, 64, 128]
NUM_WARPS_AUTOTUNE = [2, 4, 8, 16] if is_amd else [4, 8, 16, 32]


def fused_recurrent_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    inplace_final_state: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    out: torch.Tensor | None = None,
    sigmoid_beta: bool = False,
    a_log: torch.Tensor | None = None,
    g_bias: torch.Tensor | None = None,
    compute_gate: bool = False,
    lower_bound: float | None = -5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = next_power_of_2(K), min(next_power_of_2(V), 8)
    NK, NV = cdiv(K, BK), cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 1

    if compute_gate:
        a_log = a_log
    grid = (NK, NV, N * HV)
    fused_recurrent_gated_delta_rule_fwd_kernel[grid](
        q=q, k=k, v=v, g=g, beta=beta, o=o, h0=initial_state, ht=final_state,
        num_warps=num_warps, num_stages=num_stages,
    )
    return o, final_state
