#include <torch/extension.h>
#include <pybind11/pybind11.h>

#include "quant/quantize.cuh"
#include "quant/exl3_gemm.cuh"
#include "cpu/moe_mul1.h"
#include "cpu/moe_handoff.h"
#include "quant/exl3_devctx.cuh"
#include "quant/exl3_moe.cuh"

#include "parallel/context.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("exl3_moe_cpu_make_layer", &exl3_moe_cpu_make_layer, "exl3_moe_cpu_make_layer");
    m.def("exl3_moe_cpu_forward", &exl3_moe_cpu_forward, "exl3_moe_cpu_forward");
    m.def("exl3_moe_max_concurrency", &exl3_moe_max_concurrency, "exl3_moe_max_concurrency");
    m.def("exl3_moe", &exl3_moe, "exl3_moe");
    m.def("had_r_128", &had_r_128, "had_r_128");
}
