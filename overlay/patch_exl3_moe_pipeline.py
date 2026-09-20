#!/usr/bin/env python3
"""[glm53-exl3-moe-pipeline] Add an opt-in register-cut variant of the fused
`exl3_moe` decode kernel, with the geometry selected by measurement.

Task 42. The fused `exl3_moe_kernel<4,256,1>` is the single largest decode kernel
in this deployment: 49.70% (rank0) / 49.32% (rank1) of decode-step kernel time,
rising to 55.2% at T=12 and 58.1% at T=20 (task 29 receipts). Its production
launch is block 512, REG:128, SMEM:92,160 B, 1 block/SM, 16 warps/SM, 33.3%
occupancy (task 36). Task 36 also proves that occupancy gap cannot be opened at
this geometry: `512 x 128 = 65,536` registers is the entire SM register file and
`2 x 92,160 = 184,320 B > 131,072 B`.

Why the lever is a register cut and not an occupancy change
-----------------------------------------------------------
SM121 has no Tensor Memory and no `tcgen05` path (those are SM100/B200-only), so
this kernel keeps its accumulator in the register file under plain `mma.sync`.
Register pressure is therefore the binding constraint, and the 128-register
ceiling forces genuine local-memory spill traffic. Measured on **our** cubin and
our toolchain (CUDA 13.0.88, `-arch=sm_121a`), the production K4/N256 instance
carries:

    REG:128  STACK:88 B   STL:37  LDL:39

A deeper fragment pipeline (`MOE_FRAG_STAGES`) holds more `FragA`/`FragB` arrays
live in registers; a deeper shared-memory pipeline (`MOE_SH_STAGES`) holds more
`cp.async` stages in smem, which costs no registers. Trading the first for the
second is the mechanism here: at `MOE_FRAG_STAGES 1` / `MOE_SH_STAGES 8` the same
instance measures

    REG:128  STACK:40 B   STL:10  LDL:5

(as measured by the standalone compile probe; the value baked into the shipped
image is quoted below)

Measured on the shipped image (`glm53-selfbuild:e3-pipeline-f1s8`), same
revision and toolchain, per-function from the extracted cubin
(`cuobjdump -res-usage` + `nvdisasm`; the vLLM image ships no `nvdisasm`):

    stock   `exl3_moe_kernel<4,256,1>`             REG:128 STACK:88 B  STL:37 LDL:39  10,256 insts
    variant `glm53_exl3_moe_pipeline_kernel<4,256,1>` REG:128 STACK:32 B  STL:9  LDL:4   5,888 insts

Same 128 registers and the same 1 block/SM occupancy. `SMEM_MAX` (90 KiB) is a
fixed launch parameter, not stage-derived, so the dynamic smem footprint is
unchanged; `exl3_gemm_inner.cuh`'s `static_assert` bounds the deeper staging
against it and a geometry that did not fit would fail the build.

Measured effect (cluster A/B, same image, this knob the only variable):
fused-kernel device time -6.38 / -6.79 / -7.34% at T = 12 / 20 / 32, with the
rebuilt image's *stock* path validated within 0.5% of the production image
first; structured decode +5.8%, hashmap prose +12%, hard essay +2.5%. The essay
lane is **below** the task's >=5% gate and that is recorded, not rounded.

Note the two figures above are the *shipped* ones. The compile-probe numbers in
this docstring's first block are from a throwaway tree and differ slightly; do
not quote them as the image's signature.

What this overlay does
----------------------
Additive only. The stock `exl3_moe_kernel` and every stock instance are left
byte-identical:

  * copies `quant/exl3_moe_kernel.cuh` to `quant/glm53_exl3_moe_pipeline_kernel.cuh`
    with the entry point renamed, so the two live side by side with no ODR clash;
  * adds `quant/comp_units/glm53_exl3_moe_pipeline.cu` instantiating the K4/N256
    pair with the chosen `MOE_FRAG_STAGES` / `MOE_SH_STAGES`;
  * adds `quant/glm53_exl3_moe_pipeline.cuh` declaring the getters and the baked
    geometry label;
  * patches the host dispatch in `quant/exl3_moe.cu` to select the variant only
    when the knob is armed **and** the geometry matches, and to **fail closed**
    otherwise -- a silent fallback would make the arm's result uninterpretable;
  * emits a one-time boot log line naming the live geometry, so the running
    process can be audited without reading the cubin;
  * exposes `glm53_exl3_moe_pipeline_geometry()` through `bindings.cpp`.

The geometry is a **build-time** choice (it is a template argument), so the arm
is selected at runtime by `GLM53_EXL3_MOE_PIPELINE` while the numbers come from
the build args. That keeps A and B on one image: the same boot layout, the same
JIT shape hash, the same capture set, and the kernel selection as the only
independent variable.

Why the deeper smem pipeline is affordable
------------------------------------------
The dynamic smem is a fixed launch parameter (`SMEM_MAX`, 90 KiB) and
`exl3_gemm_kernel_inner` already asserts the budget fits:

    static_assert(SMEM_MAX >= SH_STAGES * (2*sh_a + 2*sh_b) + 4*sh_c)

so raising `SH_STAGES` re-partitions the same 90 KiB rather than asking for more,
and a geometry that did not fit would fail the build rather than at run time.
Raising it does not open task 36 either: the register file and the per-block smem
ceiling are unchanged, so residency stays at 1 block/SM by construction.

Usage (in the image build, after the aarch64 stub patch):
    python3 patch_exl3_moe_pipeline.py EXT_ROOT [--frag N] [--sh N]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

MARKER = "[glm53-exl3-moe-pipeline]"

GEOMETRY_HEADER_REL = "quant/glm53_exl3_moe_pipeline.cuh"
KERNEL_HEADER_REL = "quant/glm53_exl3_moe_pipeline_kernel.cuh"
COMP_UNIT_REL = "quant/comp_units/glm53_exl3_moe_pipeline.cu"

# --- anchors -----------------------------------------------------------------

KERNEL_DEF_OLD = "void exl3_moe_kernel(EXL3_MOE_KERNEL_ARGS)"
KERNEL_DEF_NEW = "void glm53_exl3_moe_pipeline_kernel(EXL3_MOE_KERNEL_ARGS)"

HOST_INCLUDE_OLD = (
    '#include "comp_units/exl3_moe_instances.cuh"\n'
    '#include "exl3_devctx.cuh"\n'
    "#include <set>\n"
)
HOST_INCLUDE_NEW = (
    '#include "comp_units/exl3_moe_instances.cuh"\n'
    '#include "exl3_devctx.cuh"\n'
    f'#include "glm53_exl3_moe_pipeline.cuh"  // {MARKER}\n'
    "#include <set>\n"
)

HOST_HELPERS_ANCHOR = "std::set<void*> moe_kernel_attr_set[MAX_DEVICES] = {};\n"

HOST_HELPERS = """
// --- [glm53-exl3-moe-pipeline] opt-in register-cut decode variant -----------
// Read once per process. The knob is the only runtime variable; the geometry is
// baked into the comp unit at build time.
static bool glm53_exl3_moe_pipeline_armed()
{
    static const bool armed = [] {
        const char* value = std::getenv("GLM53_EXL3_MOE_PIPELINE");
        TORCH_CHECK(!value || !std::strcmp(value, "0") || !std::strcmp(value, "1"),
                    "GLM53_EXL3_MOE_PIPELINE must be 0 or 1");
        return value && !std::strcmp(value, "1");
    }();
    return armed;
}

static bool glm53_exl3_moe_pipeline_device_ok(int device)
{
    TORCH_CHECK(device >= 0 && device < MAX_DEVICES, "Unexpected device index");
    static thread_local int cached[MAX_DEVICES] = {};
    if (cached[device] == 0)
    {
        int major = 0;
        int minor = 0;
        TORCH_CHECK(cudaDeviceGetAttribute(
                        &major, cudaDevAttrComputeCapabilityMajor, device) == cudaSuccess,
                    "cudaDeviceGetAttribute(major) failed");
        TORCH_CHECK(cudaDeviceGetAttribute(
                        &minor, cudaDevAttrComputeCapabilityMinor, device) == cudaSuccess,
                    "cudaDeviceGetAttribute(minor) failed");
        cached[device] = (major == 12 && minor == 1) ? 1 : -1;
    }
    return cached[device] == 1;
}

static fp_exl3_moe_kernel glm53_exl3_moe_pipeline_kernel_for(int cb_idx)
{
    return cb_idx == 0 ? glm53_exl3_moe_pipeline_k4_n256_cb1()
                       : glm53_exl3_moe_pipeline_k4_n256_cb2();
}

std::string glm53_exl3_moe_pipeline_geometry()
{
    return glm53_exl3_moe_pipeline_geometry_literal();
}
"""

HOST_DISPATCH_OLD = (
    "    fp_exl3_moe_kernel kernel = exl3_moe_kernel_instances[4 * K + 2 * cb_idx + N_off];\n"
)

HOST_DISPATCH_NEW = HOST_DISPATCH_OLD + """
    // [glm53-exl3-moe-pipeline] Swap in the register-cut variant. Fail closed:
    // this geometry is the only one the variant is instantiated for, and a
    // silent fallback to stock would leave the arm armed but inert, which would
    // make every measurement taken under it uninterpretable.
    if (glm53_exl3_moe_pipeline_armed())
    {
        TORCH_CHECK(glm53_exl3_moe_pipeline_device_ok(device),
                    "GLM53_EXL3_MOE_PIPELINE=1 requires SM121 (cc 12.1)");
        TORCH_CHECK(K == 4 && N_off == 1 && cb_idx == 0,
                    "GLM53_EXL3_MOE_PIPELINE=1 is instantiated for the K4/N256/mcg "
                    "geometry only (got K=", K, " N_off=", N_off, " cb_idx=", cb_idx,
                    "); refusing to serve with a silently inert arm");
        kernel = glm53_exl3_moe_pipeline_kernel_for(cb_idx);
        static bool logged = false;
        if (!logged)
        {
            logged = true;
            fprintf(stderr,
                    "[glm53-exl3-moe-pipeline] active: K4/N256/mcg geometry=%s "
                    "(register-cut decode)\\n",
                    glm53_exl3_moe_pipeline_geometry().c_str());
            fflush(stderr);
        }
    }
"""

BINDINGS_INCLUDE_OLD = '#include "quant/exl3_moe.cuh"\n'
BINDINGS_INCLUDE_NEW = (
    '#include "quant/exl3_moe.cuh"\n'
    f'#include "quant/glm53_exl3_moe_pipeline.cuh"  // {MARKER}\n'
)

BINDINGS_DEF_OLD = '    m.def("exl3_moe", &exl3_moe, "exl3_moe");\n'
BINDINGS_DEF_NEW = BINDINGS_DEF_OLD + (
    f'    m.def("glm53_exl3_moe_pipeline_geometry", &glm53_exl3_moe_pipeline_geometry,\n'
    f'          "glm53_exl3_moe_pipeline_geometry");  // {MARKER}\n'
)

GEOMETRY_HEADER = f"""#pragma once
// {MARKER} declarations for the opt-in register-cut fused MoE decode variant.
// Generated by overlay/patch_exl3_moe_pipeline.py -- do not edit in the image.

#include <string>

#include "comp_units/exl3_moe_instances.cuh"

fp_exl3_moe_kernel glm53_exl3_moe_pipeline_k4_n256_cb1();
fp_exl3_moe_kernel glm53_exl3_moe_pipeline_k4_n256_cb2();

// Baked geometry label, e.g. "frag1_sh8". Defined in the generated comp unit.
std::string glm53_exl3_moe_pipeline_geometry_literal();

// The label as reported to the operator (host-side wrapper in exl3_moe.cu).
std::string glm53_exl3_moe_pipeline_geometry();
"""

COMP_UNIT = """// {marker} generated comp unit -- do not edit in the image.
//
// The stock `exl3_moe_kernel` is untouched. This translation unit instantiates
// the renamed copy with a shallower fragment pipeline and a deeper shared-memory
// pipeline, trading register-resident fragment stages (which spill) for
// smem-resident cp.async stages (which do not).
//
// `exl3_moe_common.cuh` is `#pragma once`, so the `#undef`/`#define` below is
// what the renamed header actually sees.

#include "exl3_moe_instances.cuh"

#undef MOE_FRAG_STAGES
#define MOE_FRAG_STAGES {frag}
#undef MOE_SH_STAGES
#define MOE_SH_STAGES {sh}

#include "../glm53_exl3_moe_pipeline_kernel.cuh"

#include <string>

fp_exl3_moe_kernel glm53_exl3_moe_pipeline_k4_n256_cb1() {{ return glm53_exl3_moe_pipeline_kernel<4, 256, 1>; }}
fp_exl3_moe_kernel glm53_exl3_moe_pipeline_k4_n256_cb2() {{ return glm53_exl3_moe_pipeline_kernel<4, 256, 2>; }}

std::string glm53_exl3_moe_pipeline_geometry_literal()
{{
    return "frag{frag}_sh{sh}";
}}
"""


def replace_once(text: str, old: str, new: str, what: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"{what}: expected exactly one anchor, found {count}: {old.strip()[:80]!r}"
        )
    return text.replace(old, new, 1)


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".glm53-pipeline.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ext_root", help="path to exllamav3_ext")
    parser.add_argument("--frag", type=int, default=1, help="MOE_FRAG_STAGES (stock 3)")
    parser.add_argument("--sh", type=int, default=8, help="MOE_SH_STAGES (stock 3)")
    args = parser.parse_args()

    if args.frag < 1 or args.frag > 5:
        raise SystemExit(f"--frag must be 1..5 (got {args.frag})")
    # `exl3_gemm_inner.cuh` has a static_assert bounding SH_STAGES against the
    # 90 KiB dynamic smem budget; a geometry that does not fit fails the build.
    # 2 is the pipeline's own floor (`cp_async_wait<SH_STAGES - 2>`).
    if args.sh < 2 or args.sh > 16:
        raise SystemExit(f"--sh must be 2..16 (got {args.sh})")

    root = Path(args.ext_root).resolve()
    quant = root / "quant"
    host_path = quant / "exl3_moe.cu"
    bindings_path = root / "bindings.cpp"
    for required in (quant, host_path, bindings_path):
        if not required.exists():
            raise SystemExit(f"invalid extension root, missing: {required}")

    kernel_src = (quant / "exl3_moe_kernel.cuh").read_text()
    host_src = host_path.read_text()
    bindings_src = bindings_path.read_text()

    if MARKER in host_src or MARKER in bindings_src:
        raise SystemExit(
            "pipeline patch already present; apply to a clean source tree"
        )

    # --- derive the renamed kernel header ----------------------------------
    kernel_out = replace_once(
        kernel_src, KERNEL_DEF_OLD, KERNEL_DEF_NEW, "exl3_moe_kernel.cuh entry point"
    )
    kernel_out = f"// {MARKER} renamed copy; geometry comes from the comp unit.\n" + kernel_out

    # --- patch the host dispatch -------------------------------------------
    host_out = replace_once(host_src, HOST_INCLUDE_OLD, HOST_INCLUDE_NEW, "exl3_moe.cu includes")
    host_out = replace_once(
        host_out, HOST_HELPERS_ANCHOR, HOST_HELPERS_ANCHOR + HOST_HELPERS, "exl3_moe.cu helper site"
    )
    host_out = replace_once(
        host_out, HOST_DISPATCH_OLD, HOST_DISPATCH_NEW, "exl3_moe.cu kernel selection"
    )

    # --- patch bindings -----------------------------------------------------
    bindings_out = replace_once(
        bindings_src, BINDINGS_INCLUDE_OLD, BINDINGS_INCLUDE_NEW, "bindings.cpp include"
    )
    bindings_out = replace_once(
        bindings_out, BINDINGS_DEF_OLD, BINDINGS_DEF_NEW, "bindings.cpp exl3_moe def"
    )

    # --- write everything only after every anchor validated -----------------
    write_atomic(quant / Path(KERNEL_HEADER_REL).name, kernel_out)
    write_atomic(quant / Path(GEOMETRY_HEADER_REL).name, GEOMETRY_HEADER)
    write_atomic(
        quant / "comp_units" / Path(COMP_UNIT_REL).name,
        COMP_UNIT.format(marker=MARKER, frag=args.frag, sh=args.sh),
    )
    write_atomic(host_path, host_out)
    write_atomic(bindings_path, bindings_out)

    print(
        f"{MARKER} installed opt-in register-cut decode variant into {root} "
        f"(frag={args.frag} sh={args.sh}, stock was 3/3)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
