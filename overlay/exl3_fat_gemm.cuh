#pragma once

#include <torch/extension.h>

// Additive K4/MCG fat GEMM. CUDA 13 `.pragma enable_smem_spilling` is
// illegal on kernels that set MaxDynamicSharedMemorySize / use
// `extern __shared__`. Do not add that pragma here.

void exl3_fat_gemm(
    at::Tensor a,
    at::Tensor packed,
    at::Tensor out,
    at::Tensor svh,
    int64_t K,
    bool mcg,
    bool mul1);

void exl3_fat_gemm_scatter(
    at::Tensor a,
    at::Tensor packed,
    at::Tensor out,
    at::Tensor svh,
    at::Tensor token_idx,
    at::Tensor route_weight,
    int64_t K,
    bool mcg,
    bool mul1);
