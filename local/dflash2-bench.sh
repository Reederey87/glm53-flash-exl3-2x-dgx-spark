#!/usr/bin/env bash
# Candidate decode lanes, same protocol as the recorded reuse baseline:
# warmup 3-run per lane, 240 s settle, then 9-run measured round.
#
# Exit status is the contract: 0 only when every bench_decode.py invocation
# succeeded AND all three measured lanes produced a usable result. A missing or
# empty lane is a failure, not a blank cell -- the earlier version printed
# "BENCH DONE" and exited 0 after an unsupported --essay left the lane absent.
#
# Overridable for the host regression test in tests/test_dflash2_smoke_harness.py:
#   GLM53_KIT_DIR  kit root to cd into (default: the Spark kit path)
#   BENCH_DECODE   benchmark driver (default: tests/bench_decode.py)
#   BENCH_OUT      output directory (default: the smoke dir's bench/)
#   BENCH_SETTLE   settle seconds (default: 240; the host test sets 0)
set -uo pipefail

D="${GLM53_KIT_DIR:-/home/nvidia/GLM-5.3-Flash-EXL3-2x-DGX-Sparks}"
cd "$D" || exit 1
BENCH_DECODE="${BENCH_DECODE:-tests/bench_decode.py}"
OUT="${BENCH_OUT:-$(cat /tmp/dflash2-smoke-dir)/bench}"
SETTLE="${BENCH_SETTLE:-240}"
mkdir -p "$OUT" || exit 1

rc=0

# Run one lane. Logs to <out>.log so a failure is inspectable; the lane's own
# exit status decides rc.
bench() { # $1 = json path; rest = bench_decode.py args
    local out="$1"
    shift
    if python3 "$BENCH_DECODE" --out "$out" "$@" > "$out.log" 2>&1; then
        tail -3 "$out.log"
    else
        echo "FAILED: bench_decode $* (see $out.log)"
        tail -8 "$out.log"
        rc=1
    fi
}

echo "=== warmup (3 runs per lane) ==="
bench "$OUT/warm-structured.json" --phase dflash2-warm-structured --runs 3 --max-tokens 200 --structured --skip-coherence
bench "$OUT/warm-hashmap.json"    --phase dflash2-warm-hashmap    --runs 3 --max-tokens 200 --skip-coherence
bench "$OUT/warm-essay.json"      --phase dflash2-warm-essay      --runs 3 --max-tokens 200 --essay --skip-coherence

echo "=== settle ${SETTLE} s ==="
sleep "$SETTLE"

echo "=== measured (9 runs per lane) ==="
bench "$OUT/structured.json" --phase dflash2-structured --runs 9 --max-tokens 200 --structured
echo
bench "$OUT/hashmap.json"    --phase dflash2-hashmap    --runs 9 --max-tokens 200
echo
bench "$OUT/essay.json"      --phase dflash2-essay      --runs 9 --max-tokens 200 --essay

echo
echo "=== summary (median tok/s, min-max, accept) ==="
python3 - "$OUT" <<'PY'
import json
import statistics
import sys
from pathlib import Path

d = Path(sys.argv[1])
print(f"{'lane':11s} {'median':>8s} {'min':>8s} {'max':>8s} {'acc/step':>9s} {'accept':>7s} {'nan':>4s}")
missing = []
for lane in ("structured", "hashmap", "essay"):
    p = d / f"{lane}.json"
    if not p.exists():
        print(f"{lane:11s} MISSING")
        missing.append(lane)
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
            missing.append(lane)
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

if missing:
    print(f"MISSING LANES: {missing}")
    sys.exit(1)
PY
summary_rc=$?
[ "$summary_rc" -eq 0 ] || rc=1

if [ "$rc" -eq 0 ]; then
    echo "=== BENCH PASS ==="
else
    echo "=== BENCH FAIL (lane error or missing result above) ==="
fi
exit "$rc"
