#!/usr/bin/env python3
"""Stub AVX CPU targets so ExLlamaV3's extension compiles on aarch64/GB10."""

from pathlib import Path
import sys

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/exllamav3/exllamav3/exllamav3_ext")
(root / "avx2_target.cpp").write_text(
    '#include "avx2_target.h"\nbool is_avx2_supported() { return false; }\n'
)
(root / "avx512_target.cpp").write_text(
    '#include "avx512_target.h"\nbool is_avx512_supported() { return false; }\n'
)
(root / "parallel/all_reduce_cpu_avx2.cpp").write_text(
    """#include "all_reduce_cpu_avx2.h"
#include "all_reduce_cpu_avx512.h"
#include <cstdlib>
void enable_fast_fp() {}
void enable_fast_fp_avx2() {}
void perform_cpu_reduce(PGContext*, size_t, uint32_t, uint8_t*, size_t) { std::abort(); }
void perform_cpu_reduce_avx2(PGContext*, size_t, uint32_t, uint8_t*, size_t) { std::abort(); }
"""
)
(root / "parallel/all_reduce_cpu_avx512.cpp").write_text(
    """#include "all_reduce_cpu_avx512.h"
#include <cstdlib>
void enable_fast_fp_avx512() {}
void bf16_add_inplace_avx512(uint16_t*, const uint16_t*, size_t) {}
void perform_cpu_reduce_avx512(PGContext*, size_t, uint32_t, uint8_t*, size_t) { std::abort(); }
"""
)
for hdr, name in (("avx2_target.h", "avx2"), ("avx512_target.h", "avx512")):
    guard = name.upper()
    (root / hdr).write_text(
        "#pragma once\n"
        f"bool is_{name}_supported();\n"
        f"#define {guard}_TARGET\n"
        f"#define {guard}_TARGET_OPTIONAL\n"
    )
print(f"aarch64 EXL3 CPU-target stubs written in {root}")
