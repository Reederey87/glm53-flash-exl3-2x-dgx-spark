#!/usr/bin/env python3
"""Build gate: prove the TP=2 specialization is really present.

Two independent checks, because either alone is weak:

1. The Python dispatch set carries (32, 2176). This is what FlashInfer consults
   to decide whether a shape has a decode kernel -- but it is just a frozenset
   that patch_tp2.py edited, so on its own it proves only that the edit ran.

2. The installed AOT .so DIFFERS from the base .so we moved aside. FlashInfer
   ships this module precompiled and it takes precedence over the sources, so if
   the rebuild silently no-ops the sources are patched, the frozenset says the
   kernel exists, and the container fails at runtime instead of at build time.
   Comparing hashes is what makes check 1 meaningful.
"""

import hashlib
import pathlib
import sys

import flashinfer.mla._sparse_mla_sm120 as m

REQUIRED = ((32, 2176), (64, 2176))  # 32 = TP=2 shard, 64 = TP=1 (from the base)
BUILT = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/flashinfer_jit_cache/"
    "jit_cache/sparse_mla_sm120/sparse_mla_sm120.so"
)
BASE = pathlib.Path("/opt/glm53-tp2/sparse_mla_sm120.base.so")

fail = False

have = m._DECODE_DSV3_2_DISPATCH
missing = [k for k in REQUIRED if k not in have]
if missing:
    print(f"FAIL: dispatch set is missing {missing}", file=sys.stderr)
    print(f"      present at topk=2176: {sorted(k for k in have if k[1] == 2176)}", file=sys.stderr)
    fail = True
else:
    print("ok: dispatch set carries", sorted(k for k in have if k[1] == 2176))


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


if not BUILT.is_file():
    print(f"FAIL: no AOT module at {BUILT}", file=sys.stderr)
    fail = True
elif not BASE.is_file():
    print(f"FAIL: base module {BASE} missing -- cannot prove a rebuild happened", file=sys.stderr)
    fail = True
elif sha(BUILT) == sha(BASE):
    print(f"FAIL: installed AOT module is byte-identical to the base ({sha(BUILT)}).", file=sys.stderr)
    print("      The rebuild did not take effect; the new kernel is NOT compiled in.", file=sys.stderr)
    fail = True
else:
    print(f"ok: AOT module rebuilt (base {sha(BASE)} -> built {sha(BUILT)})")

sys.exit(1 if fail else 0)
