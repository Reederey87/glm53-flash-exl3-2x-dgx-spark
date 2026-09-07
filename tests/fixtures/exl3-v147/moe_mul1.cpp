#include "moe_mul1.h"
#include <c10/util/Half.h>
#include <torch/extension.h>
#include <immintrin.h>

#if defined(__GNUC__) && defined(__linux__)
#define M1_TARGET_AVX2 __attribute__((target("avx2,fma,f16c")))
#define M1_TARGET_VNNI __attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,fma,f16c")))
#else
#define M1_TARGET_AVX2
#define M1_TARGET_VNNI
#endif

// Compact v1.4.7-class x86 CPU-MoE TU for host aarch64-stub tests.
// The production pin ships the full source; this fixture only needs the
// unguarded immintrin include and AVX target attributes that break aarch64.

M1_TARGET_AVX2 void hadamard_128_avx2(float* v);
int64_t exl3_moe_cpu_make_layer(
    const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,
    int64_t, double, int64_t);
void exl3_moe_cpu_forward(int64_t, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t);
