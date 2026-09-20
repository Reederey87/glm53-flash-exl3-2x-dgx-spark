#!/usr/bin/env bash
# Candidate decode lanes, same protocol as the recorded reuse baseline:
# warmup 3-run per lane, 240 s settle, then 9-run measured round.
set -uo pipefail
D=/home/nvidia/GLM-5.3-Flash-EXL3-2x-DGX-Sparks
cd "$D"
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
import json, statistics, sys
from pathlib import Path
d = Path(sys.argv[1])
for lane in ("structured", "hashmap", "essay"):
    p = d / f"{lane}.json"
    if not p.exists():
        print(f"{lane:11s} MISSING"); continue
    rec = json.loads(p.read_text())
    runs = rec.get("runs", [])
    tps = [r.get("tok_s") for r in runs if isinstance(r.get("tok_s"), (int, float))]
    acc = [r.get("accept") for r in runs if isinstance(r.get("accept"), (int, float))]
    nan = sum(1 for r in runs if r.get("nan"))
    if not tps:
        print(f"{lane:11s} no tok_s"); continue
    med = statistics.median(tps)
    print(f"{lane:11s} median={med:7.3f} min={min(tps):7.3f} max={max(tps):7.3f} "
          f"acc={statistics.median(acc):.3f}" if acc else f"{lane:11s} median={med:.3f}",
          f"nan={nan}")
PY
echo "=== BENCH DONE ==="
