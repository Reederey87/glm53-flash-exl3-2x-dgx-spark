#!/usr/bin/env bash
# Task 42 lever 2 window: same-image A/B of GLM53_EXL3_MOE_REUSE on
# glm53-selfbuild:e3-pipeline-f1s8-reuse.
#
# Control = pipeline kernel, shared_input=false (current production behaviour).
# Variant = same cubin, shared_input=true (skip the duplicate up Hadamard).
#
# Rollback named before the arm:
#   GLM53_EXL3_MOE_REUSE=0 on this image, or IMAGE=glm53-selfbuild:e3-pipeline-f1s8
#   if the rebuild itself is not rebuild-neutral.
set -euo pipefail

KIT=/home/nvidia/GLM-5.3-Flash-EXL3-2x-DGX-Sparks
OUT=/home/nvidia/t42-reuse-window
NEW=glm53-selfbuild:e3-pipeline-f1s8-reuse
WORKER=nvidia@192.168.177.11
export SKIP_PULL=1
export SKIP_DOWNLOAD=1
export SKIP_SYNC=1
# Overlay GPU self-check ran during image build, kprobe, and the first boot
# attempt. Repeating it immediately before vLLM's fit check is the 0.25 GiB
# miss that killed the first control boot (103.16 < 103.44).
export SKIP_OVERLAY_VERIFY=1
export WORKER_SSH="$WORKER"

cd "$KIT"
mkdir -p "$OUT"

log() { printf '[t42-reuse %s] %s\n' "$(date -Is)" "$*"; }
die() { log "ERROR: $*"; exit 1; }

need_image() {
    docker image inspect "$NEW" >/dev/null 2>&1 \
        || die "$NEW is not on the head — cubin rebuild has not finished"
}

memfree_gib() {
    awk '/^MemFree:/ {printf "%d", $2/1048576}' /proc/meminfo
}
memfree_gib_worker() {
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER" \
        "awk '/^MemFree:/ {printf \"%d\", \$2/1048576}' /proc/meminfo"
}

wait_mem() {
    local need="${1:-90}" deadline
    deadline=$(( $(date +%s) + 600 ))
    while :; do
        local h w
        h="$(memfree_gib)"; w="$(memfree_gib_worker)"
        log "MemFree head=${h}GiB worker=${w}GiB (need ${need})"
        if [[ "$h" =~ ^[0-9]+$ ]] && [[ "$w" =~ ^[0-9]+$ ]] \
           && [ "$h" -ge "$need" ] && [ "$w" -ge "$need" ]; then
            return 0
        fi
        [ "$(date +%s)" -ge "$deadline" ] && { log "WARN: MemFree wait timed out"; return 1; }
        sleep 10
    done
}

arm_env() {  # $1 = 0|1
    local reuse="$1"
    # last-wins append; never edit earlier IMAGE= lines in place
    printf '\n# --- task 42 lever 2 window %s ---\nIMAGE=%s\nGLM53_EXL3_MOE_PIPELINE=1\nGLM53_EXL3_MOE_REUSE=%s\n' \
        "$(date -Is)" "$NEW" "$reuse" >> .env
    log "armed IMAGE=$NEW PIPELINE=1 REUSE=$reuse"
    grep -nE '^(IMAGE|GLM53_EXL3_MOE_PIPELINE|GLM53_EXL3_MOE_REUSE)=' .env | tail -6
    ./start.sh validate
}

wait_health() {
    local i c
    for i in $(seq 1 180); do
        c=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 http://127.0.0.1:8000/health || true)
        if [ "$c" = "200" ]; then
            log "health 200 after $((i*10))s"
            return 0
        fi
        sleep 10
    done
    die "health did not return 200"
}

assert_arm() {  # $1 expected reuse 0|1
    local expect="$1" head_n worker_n image labels
    image=$(docker inspect -f '{{.Config.Image}}' glm53-exl3-head)
    [ "$image" = "$NEW" ] || die "head image is $image, want $NEW"
    labels=$(docker inspect -f 'pipeline={{index .Config.Labels "glm53.task42.pipeline"}} reuse={{index .Config.Labels "glm53.task42.reuse"}}' "$NEW")
    echo "$labels" | grep -q 'reuse=1' || die "image reuse label missing: $labels"
    # Capture logs first. `docker logs | grep -q` under pipefail is SIGPIPE
    # on a large stream and looks like a missing arming line.
    local head_logs worker_logs
    head_logs=$(docker logs glm53-exl3-head 2>&1 || true)
    worker_logs=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER" \
        "docker logs glm53-exl3-worker 2>&1" || true)
    head_n=$(printf '%s\n' "$head_logs" | grep -c 'glm53-exl3-moe-pipeline' || true)
    worker_n=$(printf '%s\n' "$worker_logs" | grep -c 'glm53-exl3-moe-pipeline' || true)
    if [ "$head_n" -lt 1 ] || [ "$worker_n" -lt 1 ]; then
        die "pipeline arming lines head=$head_n worker=$worker_n (want >=1 on both)"
    fi
    if [ "$expect" = "1" ]; then
        printf '%s\n' "$head_logs" | grep 'shared_input=1' >/dev/null \
            || die "head missing shared_input=1 arming line"
        printf '%s\n' "$head_logs" | grep 'gate/up Hadamard reuse' >/dev/null \
            || die "head missing Hadamard-reuse arming line"
        printf '%s\n' "$worker_logs" | grep 'shared_input=1' >/dev/null \
            || die "worker missing shared_input=1"
    else
        printf '%s\n' "$head_logs" | grep 'glm53-exl3-moe-pipeline' | grep 'shared_input=0' >/dev/null \
            || die "head missing shared_input=0 (control)"
        if printf '%s\n' "$head_logs" | grep 'shared_input=1' >/dev/null; then
            die "control arm unexpectedly selected shared_input=1"
        fi
    fi
    log "arm OK: image=$image head_lines=$head_n worker_lines=$worker_n expect_reuse=$expect"
    # `head` under pipefail is SIGPIPE; swallow it so the window does not die
    # immediately after a successful boot.
    docker logs glm53-exl3-head 2>&1 | grep -E 'glm53-exl3-moe-pipeline|reuse_aliased' | head -5 || true
    curl -s http://127.0.0.1:8000/metrics 2>/dev/null \
        | grep -oE 'kv_cache_size_tokens="[0-9]+"|kv_cache_max_concurrency="[0-9.]+"|num_gpu_blocks="[0-9]+"' \
        | sort -u || true
}

boot_arm() {  # $1 reuse 0|1  $2 logfile
    local reuse="$1" logf="$2"
    ./start.sh stop || true
    wait_mem 90 || true
    arm_env "$reuse"
    log "booting REUSE=$reuse via prod-start (settle + 3-attempt memory retry)"
    # prod-start waits for MemFree and retries the 0.25 GiB fit-check miss.
    SKIP_PULL=1 SKIP_DOWNLOAD=1 SKIP_SYNC=1 SKIP_OVERLAY_VERIFY=1 \
        ./local/prod-start.sh > "$logf" 2>&1
    log "prod-start exit=$?"
    wait_health
    assert_arm "$reuse"
}

cmd="${1:-}"
case "$cmd" in
    cubin)
        need_image
        log "image labels"
        docker inspect "$NEW" --format 'Id={{.Id}} pipeline={{index .Config.Labels "glm53.task42.pipeline"}} reuse={{index .Config.Labels "glm53.task42.reuse"}} stamp={{index .Config.Labels "glm53.recipe.stamp"}}'
        docker run --rm --entrypoint python3 "$NEW" -c 'import exllamav3_ext, pathlib, shutil; p=pathlib.Path(exllamav3_ext.__file__); print(p); shutil.copy(p, "/tmp/exllamav3_ext.so"); print("copied", p.stat().st_size)' \
            > "$OUT/ext-path.txt" 2>&1 || true
        # Extract the extension from a throwaway container filesystem.
        cid=$(docker create "$NEW")
        docker cp "$cid":/usr/local/lib/python3.12/dist-packages/exllamav3_ext.cpython-312-aarch64-linux-gnu.so "$OUT/exllamav3_ext.so" \
            || true
        docker rm "$cid" >/dev/null
        if [ -f "$OUT/exllamav3_ext.so" ]; then
            log "cuobjdump -res-usage (pipeline kernels)"
            if command -v cuobjdump >/dev/null; then
                cuobjdump -res-usage "$OUT/exllamav3_ext.so" \
                    | grep -A2 -E 'glm53_exl3_moe_pipeline_kernel|exl3_moe_kernel' \
                    | tee "$OUT/cubin-res-usage.txt"
            else
                log "host has no cuobjdump; trying the image"
                docker run --rm --entrypoint bash -v "$OUT:/out" "$NEW" -lc \
                    'cuobjdump -res-usage /usr/local/lib/python3.12/dist-packages/exllamav3_ext*.so 2>/dev/null | grep -A2 -E "glm53_exl3_moe_pipeline_kernel|exl3_moe_kernel" | tee /out/cubin-res-usage.txt' \
                    || log "WARN: no cuobjdump in image either"
            fi
            if command -v nvdisasm >/dev/null; then
                nvdisasm "$OUT/exllamav3_ext.so" 2>/dev/null \
                    | grep -cE 'STL|LDL' | tee "$OUT/cubin-stl-ldl-count.txt" || true
            fi
        else
            log "WARN: could not copy extension so; listing image site-packages"
            docker run --rm --entrypoint bash "$NEW" -lc 'ls -l /usr/local/lib/python3.12/dist-packages/exllamav3_ext*'
        fi
        ;;
    kprobe)
        need_image
        docker ps --format '{{.Names}}' | grep -q glm53 && die "glm53 container still running; stop production first"
        krun() {  # $1 label $2 reuse
            local label="$1" reuse="$2"
            log "kprobe $label REUSE=$reuse"
            docker rm -f "t42k-$label" >/dev/null 2>&1 || true
            timeout 900 docker run --rm --name "t42k-$label" --gpus all --entrypoint python3 \
                -e GLM53_EXL3_MOE_PIPELINE=1 -e GLM53_EXL3_MOE_REUSE="$reuse" \
                -e EXL3_TEMP_ROWS_FUSED=32 -e EXL3_FAT_GROUPED=1 \
                -v "$OUT/kprobe.py:/kprobe.py:ro" -v "$OUT:/out" \
                "$NEW" /kprobe.py --out "/out/kprobe-$label.json" --tokens 12,20,32 --cap 32
        }
        krun ctrl 0
        krun var 1
        python3 - <<'PY'
import json, pathlib
out = pathlib.Path.home() / "t42-reuse-window"
c = json.loads((out/"kprobe-ctrl.json").read_text())
v = json.loads((out/"kprobe-var.json").read_text())
print(f"ctrl reuse={c.get('reuse')} aliased={c.get('reuse_aliased')}")
print(f"var  reuse={v.get('reuse')} aliased={v.get('reuse_aliased')}")
print(f"{'T':>4} {'ctrl fused us':>14} {'var fused us':>14} {'delta':>10}")
for cc, vv in zip(c["cases"], v["cases"]):
    d = (vv["fused_us_per_call"] - cc["fused_us_per_call"]) / cc["fused_us_per_call"] * 100
    print(f"{cc['tokens']:>4} {cc['fused_us_per_call']:>14.1f} {vv['fused_us_per_call']:>14.1f} {d:>9.2f}%")
    print("  ctrl kernels:", list(cc["fused_kernels"])[:1])
    print("  var  kernels:", list(vv["fused_kernels"])[:1])
PY
        ;;
    parity)
        need_image
        docker ps --format '{{.Names}}' | grep -q glm53 && die "glm53 container still running"
        prun() {
            local label="$1" reuse="$2"
            log "parity $label REUSE=$reuse"
            docker rm -f "t42p-$label" >/dev/null 2>&1 || true
            timeout 900 docker run --rm --name "t42p-$label" --gpus all --entrypoint python3 \
                -e GLM53_EXL3_MOE_PIPELINE=1 -e GLM53_EXL3_MOE_REUSE="$reuse" \
                -e EXL3_TEMP_ROWS_FUSED=32 -e EXL3_FAT_GROUPED=1 \
                -v "$OUT/parity.py:/parity.py:ro" -v "$OUT:/out" \
                "$NEW" /parity.py --out "/out/parity-$label.json" --tokens 12,20,32,1024 --cap 32
        }
        prun ctrl 0
        prun var 1
        docker run --rm --entrypoint python3 -v "$OUT:/out" "$NEW" /out/cmp.py /out \
            | tee "$OUT/parity-compare.txt"
        ;;
    control)
        boot_arm 0 "$OUT/boot-ctrl.log"
        ;&
    control-measure)
        if [ "$cmd" = "control-measure" ]; then
            assert_arm 0
        fi
        log "acceptance (also warms)"
        bash local/acceptance.sh 2>&1 | tee "$OUT/accept-ctrl.log" | tail -8
        log "warmup round (discarded)"
        python3 tests/bench_decode.py --phase reuse-ctrl-warm-structured \
            --out "$OUT/ctrl-warm-structured.json" --runs 3 --max-tokens 200 --structured \
            | tee "$OUT/ctrl-warm-structured.log" | grep -E "tok_s_median" || true
        python3 tests/bench_decode.py --phase reuse-ctrl-warm-hashmap \
            --out "$OUT/ctrl-warm-hashmap.json" --runs 3 --max-tokens 200 \
            | tee "$OUT/ctrl-warm-hashmap.log" | grep -E "tok_s_median" || true
        python3 scripts/probe_v149_qualification.py --kind essay --runs 3 \
            --out "$OUT/ctrl-warm-essay.json" \
            | tee "$OUT/ctrl-warm-essay.log" | tail -6
        log "settling 240s"
        sleep 240
        grep -E "MemFree|MemAvailable" /proc/meminfo
        log "MEASURED control round"
        python3 tests/bench_decode.py --phase reuse-ctrl-structured \
            --out "$OUT/ctrl-structured.json" --runs 9 --max-tokens 200 --structured \
            | tee "$OUT/ctrl-structured.log"
        python3 tests/bench_decode.py --phase reuse-ctrl-hashmap \
            --out "$OUT/ctrl-hashmap.json" --runs 9 --max-tokens 200 \
            | tee "$OUT/ctrl-hashmap.log"
        python3 scripts/probe_v149_qualification.py --kind essay --runs 9 \
            --out "$OUT/ctrl-essay.json" \
            | tee "$OUT/ctrl-essay.log"
        log "CONTROL DONE"
        ;;
    variant)
        boot_arm 1 "$OUT/boot-var.log"
        ;&
    variant-measure)
        if [ "$cmd" = "variant-measure" ]; then
            assert_arm 1
        fi
        log "acceptance (also warms)"
        bash local/acceptance.sh 2>&1 | tee "$OUT/accept-var.log" | tail -8
        log "warmup round (discarded)"
        python3 tests/bench_decode.py --phase reuse-var-warm-structured \
            --out "$OUT/var-warm-structured.json" --runs 3 --max-tokens 200 --structured \
            | tee "$OUT/var-warm-structured.log" | grep -E "tok_s_median" || true
        python3 tests/bench_decode.py --phase reuse-var-warm-hashmap \
            --out "$OUT/var-warm-hashmap.json" --runs 3 --max-tokens 200 \
            | tee "$OUT/var-warm-hashmap.log" | grep -E "tok_s_median" || true
        python3 scripts/probe_v149_qualification.py --kind essay --runs 3 \
            --out "$OUT/var-warm-essay.json" \
            | tee "$OUT/var-warm-essay.log" | tail -6
        log "settling 240s"
        sleep 240
        grep -E "MemFree|MemAvailable" /proc/meminfo
        log "MEASURED variant round"
        python3 tests/bench_decode.py --phase reuse-var-structured \
            --out "$OUT/var-structured.json" --runs 9 --max-tokens 200 --structured \
            | tee "$OUT/var-structured.log"
        python3 tests/bench_decode.py --phase reuse-var-hashmap \
            --out "$OUT/var-hashmap.json" --runs 9 --max-tokens 200 \
            | tee "$OUT/var-hashmap.log"
        python3 scripts/probe_v149_qualification.py --kind essay --runs 9 \
            --out "$OUT/var-essay.json" \
            | tee "$OUT/var-essay.log"
        log "VARIANT DONE"
        ;;
    summary)
        python3 - <<'PY'
import json, pathlib, statistics, math
out = pathlib.Path.home() / "t42-reuse-window"

def load(name):
    p = out / name
    if not p.is_file():
        return None
    return json.loads(p.read_text())

def median_tok(doc):
    if not doc:
        return None
    if "tok_s_median" in doc and doc["tok_s_median"] is not None:
        return doc["tok_s_median"]
    runs = [r.get("tok_s") for r in doc.get("runs", []) if r.get("tok_s")]
    return statistics.median(runs) if runs else None

rows = []
for lane, cname, vname in (
    ("structured", "ctrl-structured.json", "var-structured.json"),
    ("hashmap", "ctrl-hashmap.json", "var-hashmap.json"),
    ("essay", "ctrl-essay.json", "var-essay.json"),
):
    c, v = load(cname), load(vname)
    ct, vt = median_tok(c), median_tok(v)
    delta = None if ct in (None, 0) or vt is None else (vt - ct) / ct * 100
    rows.append((lane, ct, vt, delta))
    print(f"{lane:12} ctrl={ct}  var={vt}  delta={None if delta is None else f'{delta:+.2f}%'}")

print("--- kernel probe ---")
kc, kv = load("kprobe-ctrl.json"), load("kprobe-var.json")
if kc and kv:
    for cc, vv in zip(kc["cases"], kv["cases"]):
        d = (vv["fused_us_per_call"] - cc["fused_us_per_call"]) / cc["fused_us_per_call"] * 100
        print(f"T={cc['tokens']}  {cc['fused_us_per_call']:.1f} -> {vv['fused_us_per_call']:.1f} us  {d:+.2f}%")

# Gate: ≥5% e2e on hashmap AND essay, structured non-inferior.
h = next(r for r in rows if r[0] == "hashmap")[3]
e = next(r for r in rows if r[0] == "essay")[3]
s = next(r for r in rows if r[0] == "structured")[3]
ok = (h is not None and e is not None and s is not None
      and h >= 5.0 and e >= 5.0 and s >= -0.5)
print("--- gate ---")
print(f"structured non-inferior: {s}")
print(f"hashmap >=5%: {h}")
print(f"essay   >=5%: {e}")
print("GATE PASS" if ok else "GATE FAIL — revert GLM53_EXL3_MOE_REUSE=0")
PY
        ;;
    revert)
        log "reverting to REUSE=0 on $NEW (keep the rebuild, disable the skip)"
        printf '\n# --- task 42 lever 2 REVERT %s ---\nIMAGE=%s\nGLM53_EXL3_MOE_PIPELINE=1\nGLM53_EXL3_MOE_REUSE=0\n' \
            "$(date -Is)" "$NEW" >> .env
        ./start.sh validate
        ./start.sh stop || true
        wait_mem 90 || true
        SKIP_PULL=1 SKIP_DOWNLOAD=1 SKIP_SYNC=1 ./start.sh start | tee "$OUT/boot-revert.log"
        wait_health
        assert_arm 0
        systemctl --user start vllm-glm53exl3-watchdog.timer
        log "reverted; watchdog re-armed"
        ;;
    adopt)
        log "adopting REUSE=1 on $NEW"
        grep -nE '^(IMAGE|GLM53_EXL3_MOE_PIPELINE|GLM53_EXL3_MOE_REUSE)=' .env | tail -6
        [ "$(sed -n 's/^GLM53_EXL3_MOE_REUSE=//p' .env | tail -1)" = "1" ] \
            || die "REUSE is not last-wins 1; run variant first or append it"
        systemctl --user reset-failed vllm-glm53exl3.service || true
        # The variant boot already has production serving the adopted knob.
        # Mark the unit in-sync with the running containers and re-arm watchdog.
        systemctl --user start vllm-glm53exl3.service || true
        systemctl --user start vllm-glm53exl3-watchdog.timer
        log "adopted; watchdog re-armed"
        ;;
    *)
        echo "usage: $0 cubin|kprobe|parity|control|control-measure|variant|variant-measure|summary|adopt|revert"
        exit 2
        ;;
esac
