#!/usr/bin/env bash
# ============================================================================
# launch.sh — one-step launcher for GLM-5.3-Flash-FP4-Local on 2x DGX Spark
# ============================================================================
#
# This is a thin, pinned wrapper around https://github.com/eugr/spark-vllm-docker.
# It does not build a vLLM image and it does not carry a second copy of the
# serving flags: it checks the host, clones the upstream launcher at a pinned
# commit, pulls the supported B12X image, and runs the recipe in
# recipes/glm-5.3-flash-fp4-local.yaml.
#
# Usage:
#   ./launch.sh                 full setup + start (the one-step path)
#   ./launch.sh setup           pull image + download/copy model, do not start
#   ./launch.sh down            stop both ranks
#   ./launch.sh status          containers + /health
#   ./launch.sh logs [worker]   follow the head (or worker) container log
#   ./launch.sh smoke           API acceptance checks
#   ./launch.sh kv              require >= 2M tokens in the booted KV pool
#   ./launch.sh spec            read MTP acceptance from /metrics
#   ./launch.sh preflight       read-only host/cluster checks
#   ./launch.sh render          print the exact vLLM command (no launch)
#
# Configuration lives in .env (copy env.example). Caller exports win over .env.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env}"
RECIPE="$SCRIPT_DIR/recipes/glm-5.3-flash-fp4-local.yaml"
MODEL_ID="nvidia/GLM-5.3-Flash-FP4-Local"

log()  { printf '\033[1;36m[glm53-fp4]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[glm53-fp4]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[glm53-fp4]\033[0m %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'EOF'
GLM-5.3-Flash-FP4-Local on 2x NVIDIA DGX Spark — one-step launcher

  ./launch.sh                 full setup + start
  ./launch.sh setup           prepare image and model only
  ./launch.sh down            stop both ranks
  ./launch.sh status          containers + /health
  ./launch.sh logs [worker]   follow the head (or worker) container log
  ./launch.sh smoke           API acceptance checks
  ./launch.sh kv              require >= 2M tokens in the booted KV pool
  ./launch.sh spec            read MTP acceptance from /metrics
  ./launch.sh preflight       read-only host/cluster checks
  ./launch.sh render          print the exact vLLM command

Environment:
  ENV_FILE=...                use a different .env (default: ./.env)
  HOST=0.0.0.0                expose the API on the LAN (default: loopback)
  PULL_IMAGE=0                skip docker pull; use the local vllm-node-b12x
  TAIL=500                    lines of container log for `logs`

The recipe is recipes/glm-5.3-flash-fp4-local.yaml. The upstream launcher is
pinned by SPARK_VLLM_REF in .env and cloned into .vendor/spark-vllm-docker.
EOF
}

# --- environment --------------------------------------------------------------
# Caller exports must survive `. .env`. Capture the launcher-level keys before
# sourcing, then re-apply them. This is deliberately a small whitelist: it is
# not a general shell re-exec.
_CAPTURED=()
for _key in PORT MAX_MODEL_LEN GPU_MEMORY_UTILIZATION CLUSTER_NODES ETH_IF \
            IB_IF COPY_HOSTS MASTER_PORT CONTAINER_NAME HOST SPARK_VLLM_DIR \
            SPARK_VLLM_REF IMAGE_REF IMAGE_TAG HF_TOKEN PULL_IMAGE TAIL; do
    if [ -n "${!_key+x}" ]; then
        _CAPTURED+=("${_key}=${!_key}")
    fi
done

load_env() {
    if [ ! -f "$ENV_FILE" ]; then
        [ -f "$SCRIPT_DIR/env.example" ] || die "missing env.example"
        cp "$SCRIPT_DIR/env.example" "$ENV_FILE"
        log "wrote $ENV_FILE from env.example — edit CLUSTER_NODES/ETH_IF/IB_IF"
    fi
    set -a
    # shellcheck disable=SC1090
    . "$ENV_FILE"
    set +a
    local _pair
    if [ "${#_CAPTURED[@]}" -gt 0 ]; then
        for _pair in "${_CAPTURED[@]}"; do
            export "${_pair?}"
        done
    fi

    : "${SPARK_VLLM_DIR:=$SCRIPT_DIR/.vendor/spark-vllm-docker}"
    case "$SPARK_VLLM_DIR" in
        /*) ;;
        *) SPARK_VLLM_DIR="$SCRIPT_DIR/$SPARK_VLLM_DIR" ;;
    esac
    : "${SPARK_VLLM_REF:=0bdc9c94cd89843e1c8f7d7b33df2d5bc746c03d}"
    : "${IMAGE_REF:=eugr/spark-vllm-b12x:latest}"
    : "${IMAGE_TAG:=vllm-node-b12x}"
    : "${CONTAINER_NAME:=vllm_node}"
    : "${PORT:=8000}"
    : "${HOST:=127.0.0.1}"
    : "${MAX_MODEL_LEN:=500000}"
    : "${GPU_MEMORY_UTILIZATION:=0.87}"
    : "${MASTER_PORT:=29501}"
    : "${PULL_IMAGE:=1}"
    : "${TAIL:=200}"
    : "${CLUSTER_NODES:=}"
    : "${ETH_IF:=}"
    : "${IB_IF:=}"
    : "${COPY_HOSTS:=}"
    : "${HF_TOKEN:=}"

    case "$TAIL" in
        ''|*[!0-9]*) die "TAIL must be a non-negative integer, got '$TAIL'" ;;
    esac
    [ -f "$RECIPE" ] || die "missing recipe: $RECIPE"
}

# --- upstream launcher --------------------------------------------------------
ensure_upstream() {
    command -v git >/dev/null 2>&1 || die "git is required"
    if [ ! -e "$SPARK_VLLM_DIR/.git" ]; then
        [ -e "$SPARK_VLLM_DIR" ] && die "$SPARK_VLLM_DIR exists but is not a git checkout"
        mkdir -p "$(dirname "$SPARK_VLLM_DIR")"
        log "cloning spark-vllm-docker into $SPARK_VLLM_DIR"
        git clone --quiet https://github.com/eugr/spark-vllm-docker.git "$SPARK_VLLM_DIR" \
            || die "could not clone spark-vllm-docker"
    fi
    if ! git -C "$SPARK_VLLM_DIR" rev-parse --verify --quiet "${SPARK_VLLM_REF}^{commit}" >/dev/null; then
        log "fetching $SPARK_VLLM_REF"
        git -C "$SPARK_VLLM_DIR" fetch --quiet origin || die "git fetch failed"
    fi
    git -C "$SPARK_VLLM_DIR" rev-parse --verify --quiet "${SPARK_VLLM_REF}^{commit}" >/dev/null \
        || die "pinned upstream ref not found: $SPARK_VLLM_REF"
    local want have
    want="$(git -C "$SPARK_VLLM_DIR" rev-parse "${SPARK_VLLM_REF}^{commit}")"
    have="$(git -C "$SPARK_VLLM_DIR" rev-parse HEAD)"
    if [ "$want" != "$have" ]; then
        log "checking out spark-vllm-docker $SPARK_VLLM_REF"
        git -C "$SPARK_VLLM_DIR" checkout --quiet --detach "$SPARK_VLLM_REF" \
            || die "could not check out $SPARK_VLLM_REF"
    fi
    [ -x "$SPARK_VLLM_DIR/run-recipe.sh" ] || die "upstream run-recipe.sh is missing"
    log "upstream: $(git -C "$SPARK_VLLM_DIR" rev-parse --short HEAD)"
}

node_list() {
    local IFS=','
    # shellcheck disable=SC2206
    local nodes=($CLUSTER_NODES)
    printf '%s\n' "${nodes[@]}"
}

ensure_nodes() {
    case "$CLUSTER_NODES" in
        *192.0.2.10*|*192.0.2.11*)
            [ "${ALLOW_EXAMPLE_NODES:-0}" = "1" ] \
                || die "CLUSTER_NODES still holds the env.example addresses — edit .env"
            ;;
    esac
    if [ -z "${CLUSTER_NODES// /}" ]; then
        log "CLUSTER_NODES is empty; running upstream discovery (interactive)"
        "$SPARK_VLLM_DIR/run-recipe.sh" --discover --config "$ENV_FILE" \
            || die "cluster discovery failed"
        load_env
    fi
    [ -n "${CLUSTER_NODES// /}" ] || die "no cluster nodes configured"
}

# --- preflight ----------------------------------------------------------------
_PREFLIGHT_FAIL=0
pf_ok()   { printf '  \033[1;32mok\033[0m   %s\n' "$*"; }
pf_warn() { printf '  \033[1;33mwarn\033[0m %s\n' "$*" >&2; }
pf_fail() { printf '  \033[1;31mFAIL\033[0m %s\n' "$*" >&2; _PREFLIGHT_FAIL=1; }

preflight() {
    _PREFLIGHT_FAIL=0
    log "preflight (read-only)"

    case "$(uname -m)" in
        aarch64|arm64) pf_ok "architecture $(uname -m)" ;;
        *) pf_fail "architecture $(uname -m); DGX Spark is aarch64" ;;
    esac

    if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
        pf_ok "docker $(docker version --format '{{.Server.Version}}' 2>/dev/null || true)"
    else
        pf_fail "docker is not installed or the daemon is not reachable"
    fi

    if command -v python3 >/dev/null 2>&1; then
        pf_ok "python3 $(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
        if python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
            pf_ok "python3 >= 3.10"
        else
            pf_fail "python3 >= 3.10 is required by the upstream recipe runner"
        fi
        if python3 -c 'import yaml' >/dev/null 2>&1; then
            pf_ok "PyYAML present"
        else
            pf_warn "PyYAML missing; upstream run-recipe.sh will try to pip install it"
        fi
    else
        pf_fail "python3 is required by the upstream recipe runner"
    fi

    if command -v nvidia-smi >/dev/null 2>&1 \
        && nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -qi 'GB10'; then
        pf_ok "NVIDIA GB10 visible"
    else
        pf_fail "NVIDIA GB10 not visible through nvidia-smi"
    fi

    local cache_dir="${HF_HOME:-$HOME/.cache/huggingface}"
    local avail_kb
    avail_kb="$(df -Pk "$HOME" 2>/dev/null | awk 'NR==2 {print $4}')"
    if [ -n "${avail_kb:-}" ]; then
        if [ "$avail_kb" -lt 188743680 ]; then
            pf_fail "less than 180 GiB free under $HOME (checkpoint is ~166 GiB + image)"
        elif [ "$avail_kb" -lt 235929600 ]; then
            pf_warn "less than 225 GiB free under $HOME; the checkpoint and image may not both fit"
        else
            pf_ok "free space under $HOME: $((avail_kb / 1048576)) GiB"
        fi
        if [ -d "$cache_dir" ]; then
            pf_ok "HF cache: $cache_dir"
        else
            pf_warn "HF cache not created yet: $cache_dir"
        fi
    fi

    local -a nodes=()
    local _node
    while IFS= read -r _node; do
        [ -n "$_node" ] && nodes+=("$_node")
    done < <(node_list 2>/dev/null || true)
    if [ "${#nodes[@]}" -gt 1 ]; then
        if ssh -o BatchMode=yes -o ConnectTimeout=8 "${nodes[1]}" true >/dev/null 2>&1; then
            pf_ok "passwordless ssh to worker ${nodes[1]}"
        else
            pf_fail "cannot ssh to worker ${nodes[1]} (set up passwordless SSH)"
        fi
    else
        pf_warn "only one node configured; this recipe needs two"
    fi

    if [ -n "${IB_IF// /}" ]; then
        local dev found=0
        for dev in ${IB_IF//,/ }; do
            if [ -e "/sys/class/infiniband/$dev" ]; then
                found=1
            else
                pf_warn "RDMA device not present: $dev"
            fi
        done
        if [ "$found" = 1 ]; then
            pf_ok "RDMA device(s) present: $IB_IF"
        fi
    else
        pf_warn "IB_IF is empty; upstream will autodetect the RoCE devices"
    fi

    if [ -r /proc/sys/vm/min_free_kbytes ]; then
        local mfk
        mfk="$(cat /proc/sys/vm/min_free_kbytes)"
        if [ "$mfk" -ge 1048576 ]; then
            pf_ok "vm.min_free_kbytes=$mfk"
        else
            pf_warn "vm.min_free_kbytes=$mfk; this pair is normally pinned to 1048576"
        fi
    fi

    if [ "$_PREFLIGHT_FAIL" -ne 0 ]; then
        die "preflight failed — fix the FAIL lines above"
    fi
    log "preflight passed"
}

# --- image --------------------------------------------------------------------
pull_image() {
    if [ "$PULL_IMAGE" != "1" ]; then
        log "PULL_IMAGE=0; using local image $IMAGE_TAG"
        docker image inspect "$IMAGE_TAG" >/dev/null 2>&1 \
            || die "local image $IMAGE_TAG is missing; set PULL_IMAGE=1 or build it"
        return
    fi
    log "pulling $IMAGE_REF"
    docker pull "$IMAGE_REF" || die "docker pull failed"
    docker tag "$IMAGE_REF" "$IMAGE_TAG" || die "docker tag failed"
    local digest
    digest="$(docker image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$IMAGE_REF" 2>/dev/null | head -1)"
    [ -n "$digest" ] || digest="$(docker image inspect --format '{{.Id}}' "$IMAGE_TAG")"
    mkdir -p "$SCRIPT_DIR/local"
    printf '%s\n' "$digest" > "$SCRIPT_DIR/local/image-digest.txt"
    log "image: $digest"
}

# --- recipe -------------------------------------------------------------------
recipe_args() {
    local mode="$1"
    shift
    local -a recipe_cmd=("$SPARK_VLLM_DIR/run-recipe.sh" "$RECIPE" --config "$ENV_FILE")
    case "$mode" in
        up)       recipe_cmd+=(--setup --daemon) ;;
        build)    recipe_cmd+=(--setup --build-only) ;;
        download) recipe_cmd+=(--download-only) ;;
        render)   recipe_cmd+=(--dry-run) ;;
        *)        die "internal: bad recipe mode $mode" ;;
    esac
    recipe_cmd+=(--port "$PORT" --host "$HOST" --max-model-len "$MAX_MODEL_LEN" \
          --gpu-mem "$GPU_MEMORY_UTILIZATION")
    if [ -n "$HF_TOKEN" ]; then
        recipe_cmd+=(-e "HF_TOKEN=$HF_TOKEN")
    fi
    recipe_cmd+=("$@")
    printf '%s\0' "${recipe_cmd[@]}"
}

run_recipe() {
    local mode="$1"
    shift
    local -a recipe_cmd=()
    local _item
    while IFS= read -r -d '' _item; do
        recipe_cmd+=("$_item")
    done < <(recipe_args "$mode" "$@")
    log "running pinned upstream recipe ($mode)"
    exec "${recipe_cmd[@]}"
}

prepare_all() {
    local -a build_cmd=()
    local -a download_cmd=()
    local _item
    while IFS= read -r -d '' _item; do
        build_cmd+=("$_item")
    done < <(recipe_args build)
    while IFS= read -r -d '' _item; do
        download_cmd+=("$_item")
    done < <(recipe_args download)
    log "stage 1/2: image on both nodes"
    "${build_cmd[@]}" || die "image preparation failed"
    log "stage 2/2: checkpoint on both nodes"
    exec "${download_cmd[@]}"
}

# --- operations ---------------------------------------------------------------
stop_all() {
    "$SPARK_VLLM_DIR/launch-cluster.sh" --config "$ENV_FILE" stop
}

status_all() {
    "$SPARK_VLLM_DIR/launch-cluster.sh" --config "$ENV_FILE" status || true
    if curl -fsS --max-time 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        log "health: ok  http://127.0.0.1:$PORT/health"
    else
        warn "health: not answering on http://127.0.0.1:$PORT/health"
    fi
}

logs_all() {
    local target="${1:-head}"
    case "$target" in
        head)
            docker logs --tail "$TAIL" -f "$CONTAINER_NAME"
            ;;
        worker)
            local -a nodes=()
            local _node
            while IFS= read -r _node; do
                [ -n "$_node" ] && nodes+=("$_node")
            done < <(node_list)
            [ "${#nodes[@]}" -gt 1 ] || die "no worker configured"
            # shellcheck disable=SC2029  # TAIL is validated numeric; CONTAINER_NAME is local config
            ssh "${nodes[1]}" "docker logs --tail $TAIL -f $CONTAINER_NAME"
            ;;
        *) die "usage: ./launch.sh logs [head|worker]" ;;
    esac
}

smoke() {
    python3 "$SCRIPT_DIR/scripts/smoke.py" \
        --base "http://127.0.0.1:$PORT" --model "$MODEL_ID" "$@"
}

kv_check() {
    python3 "$SCRIPT_DIR/scripts/verify_kv_pool.py" \
        --container "$CONTAINER_NAME" --min-tokens 2000000 "$@"
}

spec_check() {
    python3 "$SCRIPT_DIR/scripts/spec_acceptance.py" \
        --base "http://127.0.0.1:$PORT" "$@"
}

# --- main ---------------------------------------------------------------------
action="${1:-up}"
shift || true

case "$action" in
    -h|--help|help) usage; exit 0 ;;
esac

load_env

case "$action" in
    up|start)
        preflight
        ensure_upstream
        ensure_nodes
        pull_image
        run_recipe up "$@"
        ;;
    setup|download)
        ensure_upstream
        ensure_nodes
        pull_image
        prepare_all
        ;;
    down|stop)
        ensure_upstream
        stop_all
        ;;
    status)
        ensure_upstream
        status_all
        ;;
    logs)
        ensure_upstream
        logs_all "$@"
        ;;
    smoke)
        smoke "$@"
        ;;
    kv)
        kv_check "$@"
        ;;
    spec)
        spec_check "$@"
        ;;
    preflight)
        preflight
        ;;
    render|dry-run)
        ensure_upstream
        ensure_nodes
        run_recipe render "$@"
        ;;
    *)
        usage >&2
        die "unknown command: $action"
        ;;
esac
