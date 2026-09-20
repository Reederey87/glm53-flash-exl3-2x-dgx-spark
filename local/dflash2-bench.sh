#!/usr/bin/env bash
# Candidate decode lanes, same protocol as the recorded reuse baseline:
# warmup 3-run per lane, 240 s settle, then 9-run measured round.
set -uo pipefail
D=/home/nvidia/GLM-5.3-Flash-EXL3-2x-DGX-Sparks
cd "$D" || exit 1
OUT=$(cat /tmp/dflash2-smoke-dir)
mkdir -p "$OUT/bench"

echo "=== warmup (3 runs per lane) ==="
python3 tests/bench_decode.py --phase dflash2-warm-structured --out "$OUT/bench/warm-structured.json" --runs 3 --max-tokens 200 --structured --skip-coherence 2>&1 | tail -3
python3 tests/bench_decode.py --phase dflash2-warm-hashmap --out "$OUT/bench/warm-hashmap.json" --runs 3 --max-tokens 200 --skip-coherence 2>&1 | tail -3
python3 tests/bench_decode.py --phase dflash2-warm-essay --out "$OUT/bench/warm-essay.json" --runs 3 --max-tokens 200 --essay --skip-coherence 2>&1 | tail -3

echo "=== settle 240 s ==="
sleep 240

echo "=== measured (9 runs per lane) ==="
python3 tests/bench_decode.py --phase dflash2-structured --out "$OUT/bench/structured.json" --runs 9 --max-tokens 200 --structured 2>&1 | tail -14
echo
python3 tests/bench_decode.py --phase dflash2-hashmap --out "$OUT/bench/hashmap.json" --runs 9 --max-tokens 200 2>&1 | tail -14
echo
python3 tests/bench_decode.py --phase dflash2-essay --out "$OUT/bench/essay.json" --runs 9 --max-tokens 200 --essay 2>&1 | tail -14

echo
echo "=== summary (median tok/s, min-max, accept) ==="
python3 - "$OUT/bench" <<'PY'
import json
import statistics
import sys
from pathlib import Path

d = Path(sys.argv[1])
print(f"{'lane':11s} {'median':>8s} {'min':>8s} {'max':>8s} {'acc/step':>9s} {'accept':>7s} {'nan':>4s}")
for lane in ("structured", "hashmap", "essay"):
    p = d / f"{lane}.json"
    if not p.exists():
        print(f"{lane:11s} MISSING")
        continue
    rec = json.loads(p.read_text())
    runs = rec.get("runs", [])

    # bench_decode.py emits the aggregates at top level; the per-run acceptance
    # is nested under runs[*].spec, NOT runs[*].accept.
    med, lo, hi = rec.get("tok_s_median"), rec.get("tok_s_min"), rec.get("tok_s_max")
    if med is None:
        tps = [r["tok_s"] for r in runs if isinstance(r.get("tok_s"), (int, float))]
        if not tps:
            print(f"{lane:11s} no tok_s")
            continue
        med, lo, hi = statistics.median(tps), min(tps), max(tps)

    acc = rec.get("accept_ratio_median")
    if acc is None:
        accs = [r.get("spec", {}).get("accept_ratio") for r in runs]
        accs = [a for a in accs if isinstance(a, (int, float))]
        acc = statistics.median(accs) if accs else None

    aps = rec.get("accepted_per_step_median")
    nan = rec.get("any_nan", sum(1 for r in runs if r.get("nan")))

    def fmt(value, spec):
        return format(value, spec) if isinstance(value, (int, float)) else "n/a"

    print(
        f"{lane:11s} {fmt(med, '8.3f')} {fmt(lo, '8.3f')} {fmt(hi, '8.3f')} "
        f"{fmt(aps, '9.3f')} {fmt(acc, '7.3f')} {str(nan):>4s}"
    )
PY
echo "=== BENCH DONE ==="
