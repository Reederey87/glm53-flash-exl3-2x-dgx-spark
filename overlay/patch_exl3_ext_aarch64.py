#!/usr/bin/env python3
"""Stub AVX CPU targets so ExLlamaV3's extension compiles on aarch64/GB10.

v1.4.7 also ships an unguarded x86 CPU-MoE TU (`cpu/moe_mul1.cpp`) that
includes `<immintrin.h>` and AVX target attributes. setup.py recursively
compiles every `.cpp`, so the previous AVX-only stubs still leave that file
in the build. Replace it with an ABI-compatible aarch64 stub before compile.

v1.4.7 `cpu/moe_handoff.cu` and `parallel/all_reduce_cpu.cu` still compile
on aarch64 except for `__builtin_ia32_pause` / `_mm_pause`. Rewrite those
to `std::this_thread::yield()` and fail closed if any leftover remains.
"""

from pathlib import Path
import sys

MOE_MUL1_STUB = """\
#include "moe_mul1.h"

#include <c10/util/Half.h>
#include <torch/extension.h>

#include <cstdint>
#include <stdexcept>

/* GLM53_AARCH64_CPU_MOE_STUB */

void exl3_moe_cpu_set_prof(bool) {}

bool exl3_moe_cpu_has_avx2() { return false; }
bool exl3_moe_cpu_has_avx512_vnni() { return false; }
bool exl3_moe_cpu_has_avx512_vbmi() { return false; }

static void _cpu_moe_unavailable()
{
    throw std::runtime_error(
        "exl3 CPU MoE is unavailable on aarch64/GB10; fused GPU exl3_moe is the serving path"
    );
}

int64_t exl3_moe_cpu_make_layer
(
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
    int64_t,
    double,
    int64_t
)
{
    _cpu_moe_unavailable();
    return -1;
}

void exl3_moe_cpu_free_layer(int64_t) {}

void exl3_moe_cpu_forward
(
    int64_t,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    at::Tensor&,
    int64_t
)
{
    _cpu_moe_unavailable();
}

void exl3_moe_cpu_forward_raw
(
    int64_t,
    const at::Half*,
    const int32_t*,
    const at::Half*,
    float*,
    int,
    int,
    int
)
{
    _cpu_moe_unavailable();
}

void exl3_moe_cpu_stage_experts
(
    int64_t,
    const uint32_t*,
    int,
    uint8_t*,
    int
)
{
    _cpu_moe_unavailable();
}

int64_t exl3_moe_cpu_pool_stress(int, int, int, int)
{
    return 0;
}
"""


def stub_avx_cpu_targets(root: Path) -> None:
    # Match v1.4.7 header APIs. Returning false is correct on aarch64: GLM
    # serving uses NCCL, not ExLlamaV3 native-TP CPU all-reduce.
    (root / "avx2_target.h").write_text(
        "#pragma once\n"
        "bool is_avx2_supported();\n"
        "bool is_f16c_supported();\n"
        "#define AVX2_TARGET\n"
        "#define AVX2_F16C_TARGET\n"
        "#define AVX2_TARGET_OPTIONAL\n"
    )
    (root / "avx512_target.h").write_text(
        "#pragma once\n"
        "bool is_avx512_supported();\n"
        "#define AVX512_TARGET\n"
        "#define AVX512_TARGET_OPTIONAL\n"
    )
    (root / "avx2_target.cpp").write_text(
        '#include "avx2_target.h"\n'
        "bool is_avx2_supported() { return false; }\n"
        "bool is_f16c_supported() { return false; }\n"
    )
    (root / "avx512_target.cpp").write_text(
        '#include "avx512_target.h"\nbool is_avx512_supported() { return false; }\n'
    )
    (root / "parallel").mkdir(parents=True, exist_ok=True)
    (root / "parallel/all_reduce_cpu_avx2.cpp").write_text(
        """#include "all_reduce_cpu_avx2.h"
#include "all_reduce_cpu_avx512.h"
#include <cstdint>
#include <cstdlib>
void enable_fast_fp() {}
void enable_fast_fp_avx2() {}
void cpu_reduce_parallel(
    void (*)(uint16_t*, const uint16_t*, const uint16_t*, size_t),
    void (*)(uint16_t*, const uint16_t*, size_t),
    uint16_t*, const uint16_t*, const uint16_t*, size_t, int
) { std::abort(); }
void perform_cpu_reduce(PGContext*, size_t, uint32_t, uint32_t, uint8_t*, size_t) { std::abort(); }
void perform_cpu_reduce_avx2(PGContext*, size_t, uint32_t, uint32_t, uint8_t*, size_t) { std::abort(); }
"""
    )
    (root / "parallel/all_reduce_cpu_avx512.cpp").write_text(
        """#include "all_reduce_cpu_avx512.h"
#include <cstdint>
#include <cstdlib>
void enable_fast_fp_avx512() {}
void bf16_add_inplace_avx512(uint16_t*, const uint16_t*, size_t) {}
void bf16_add_twosrc_avx512(uint16_t*, const uint16_t*, const uint16_t*, size_t) {}
void fp16_add_inplace_avx512(uint16_t*, const uint16_t*, size_t) {}
void fp16_add_twosrc_avx512(uint16_t*, const uint16_t*, const uint16_t*, size_t) {}
void perform_cpu_reduce_avx512(PGContext*, size_t, uint32_t, uint32_t, uint8_t*, size_t) { std::abort(); }
"""
    )


def stub_cpu_moe_mul1(root: Path) -> str:
    """Replace the x86 CPU-MoE TU when present. Returns the action taken."""
    path = root / "cpu" / "moe_mul1.cpp"
    if not path.is_file():
        return "absent"
    current = path.read_text()
    if "GLM53_AARCH64_CPU_MOE_STUB" in current:
        return "already"
    if "<immintrin.h>" not in current and "M1_TARGET_AVX2" not in current:
        raise SystemExit(
            f"aarch64 stub FATAL: {path} is not the v1.4.7 x86 CPU-MoE TU "
            "(no immintrin.h / AVX target) and is not this kit's stub — refusing"
        )
    path.write_text(MOE_MUL1_STUB)
    if "<immintrin.h>" in path.read_text():
        raise SystemExit(f"aarch64 stub FATAL: post-write {path} still includes immintrin.h")
    return "stubbed"


X86_PAUSE_MARKERS = (b"__builtin_ia32_pause", b"_mm_pause")
X86_ISA_MARKERS = (
    b"immintrin.h",
    b"__builtin_ia32_pause",
    b"_mm_pause",
    b'__attribute__((target("avx',
    b'target_clones("avx',
    b"__builtin_cpu_supports",
)
PORTABLE_PAUSE = "std::this_thread::yield(); /* GLM53_AARCH64_CPU_PAUSE_STUB */"


def remaining_x86_isa(root: Path) -> list[tuple[Path, str]]:
    hits: list[tuple[Path, str]] = []
    for path in root.rglob("*"):
        if path.suffix not in {".c", ".cpp", ".h", ".hpp", ".cu", ".cuh"}:
            continue
        data = path.read_bytes()
        for marker in X86_ISA_MARKERS:
            if marker in data:
                hits.append((path, marker.decode("ascii", "replace")))
                break
    return hits

def remaining_immintrin(root: Path) -> list[Path]:
    return [path for path, marker in remaining_x86_isa(root) if marker == "immintrin.h"]


def remaining_x86_pauses(root: Path) -> list[Path]:
    hits: list[Path] = []
    for path in root.rglob("*"):
        if path.suffix not in {".c", ".cpp", ".h", ".hpp", ".cu", ".cuh"}:
            continue
        data = path.read_bytes()
        if any(marker in data for marker in X86_PAUSE_MARKERS):
            hits.append(path)
    return hits


def rewrite_x86_pauses(root: Path) -> list[str]:
    """Replace leftover x86 pause builtins. Returns rewritten relative paths."""
    rewritten: list[str] = []
    for path in remaining_x86_pauses(root):
        text = path.read_text()
        if PORTABLE_PAUSE in text and not any(m.decode() in text for m in X86_PAUSE_MARKERS):
            continue
        new = text.replace("__builtin_ia32_pause();", PORTABLE_PAUSE)
        new = new.replace("_mm_pause();", PORTABLE_PAUSE)
        if new == text:
            raise SystemExit(
                f"aarch64 stub FATAL: {path} has an x86 pause form this installer "
                "does not rewrite — refusing"
            )
        if "#include <thread>" not in new:
            new = '#include <thread>\n' + new
        path.write_text(new)
        rewritten.append(str(path.relative_to(root)))
    leftover = remaining_x86_pauses(root)
    if leftover:
        raise SystemExit(
            "aarch64 stub FATAL: x86 pause builtins still present after rewrite: "
            + ", ".join(str(p.relative_to(root)) for p in leftover)
        )
    return rewritten


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/exllamav3/exllamav3/exllamav3_ext")
    stub_avx_cpu_targets(root)
    moe_state = stub_cpu_moe_mul1(root)
    leftover = remaining_immintrin(root)
    if leftover:
        raise SystemExit(
            "aarch64 stub FATAL: immintrin.h still present after stubbing: "
            + ", ".join(str(p.relative_to(root)) for p in leftover)
        )
    pauses = rewrite_x86_pauses(root)
    leftover_isa = remaining_x86_isa(root)
    if leftover_isa:
        raise SystemExit(
            "aarch64 stub FATAL: leftover x86 ISA after stubbing: "
            + ", ".join(f"{p.relative_to(root)}:{m}" for p, m in leftover_isa)
        )
    avx2_h = (root / "avx2_target.h").read_text()
    avx2_cpp = (root / "avx2_target.cpp").read_text()
    if "is_f16c_supported" not in avx2_h or "bool is_f16c_supported() { return false; }" not in avx2_cpp:
        raise SystemExit("aarch64 stub FATAL: is_f16c_supported missing from AVX2 stubs")
    pause_note = ",".join(pauses) if pauses else "none"
    print(
        f"aarch64 EXL3 CPU-target stubs written in {root} "
        f"(cpu_moe={moe_state}, x86_pause={pause_note}, f16c=stubbed)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
