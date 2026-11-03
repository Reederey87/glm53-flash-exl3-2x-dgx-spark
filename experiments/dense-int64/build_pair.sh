#!/bin/bash
# Run only inside the generated source context on an idle Spark head.
# This builds isolated tags; it never stops, retags or starts production.
set -euo pipefail
cd -- "$(dirname -- "$0")"
tag_prefix=${1:?Usage: bash build_pair.sh glm53-int64-UNIQUE-LABEL}
[[ "$tag_prefix" =~ ^glm53-int64-[a-z0-9][a-z0-9_.-]*$ ]] || { echo 'Invalid experiment tag prefix' >&2; exit 2; }
[[ $(id -un) == nvidia && $(uname -m) == aarch64 ]] || { echo 'Requires cluster user nvidia on the ARM64 head' >&2; exit 2; }
[[ ! -e artifacts ]] || { echo 'Artifacts already exist; inspect the prior run instead of overwriting it' >&2; exit 2; }
sha256sum --quiet -c context-sha256.txt
for arm in control candidate; do
    if docker image inspect "$tag_prefix-$arm" >/dev/null 2>&1; then
        echo "Refusing to replace existing image $tag_prefix-$arm" >&2
        exit 2
    fi
done

check_idle() {
    local free_kib
    [[ -z $(docker ps -q --filter name=vllm_node) ]] || { echo 'Head serving container still running' >&2; return 1; }
    free_kib=$(awk '/^MemFree:/ {print $2}' /proc/meminfo)
    [[ "$free_kib" =~ ^[0-9]+$ && "$free_kib" -ge 94371840 ]] || { echo "Head MemFree below 90 GiB: $free_kib KiB" >&2; return 1; }
    ssh -n -T -o BatchMode=yes -o ConnectTimeout=10 nvidia@192.168.177.11 '
        set -eu
        test -z "$(docker ps -q --filter name=vllm_node)"
        free_kib=$(awk '\''/^MemFree:/ {print $2}'\'' /proc/meminfo)
        test "$free_kib" -ge 94371840
        printf "Worker MemFree: %s KiB\n" "$free_kib"
    '
    printf 'Head MemFree: %s KiB\n' "$free_kib"
}
if systemctl --user is-active --quiet vllm-glm53fp4.service ||
   systemctl --user is-active --quiet vllm-glm53fp4-watchdog.timer; then
    echo 'Stop the serving unit and disarm its watchdog through the registered window first' >&2
    exit 2
fi
check_idle
mkdir -p artifacts
exec > >(tee -a artifacts/build.log) 2>&1
common=(--platform linux/arm64 --progress plain
    --build-arg BUILD_JOBS=4 --build-arg TORCH_CUDA_ARCH_LIST=12.1a
    --build-arg FLASHINFER_CUDA_ARCH_LIST=12.1a
    --build-arg TORCH_VERSION=2.13.0 --build-arg TORCHVISION_VERSION=0.28.0
    --build-arg TORCHAUDIO_VERSION=2.11.0 --build-arg CUTLASS_DSL_VERSION=4.7.0)
run_build() {
    check_idle
    printf '%q ' docker build "${common[@]}" "$@"
    printf '\n'
    docker build "${common[@]}" "$@"
}

# Exactly one freshly compiled vLLM wheel is shared by both image arms.
run_build --target vllm-export --output type=local,dest=artifacts/vllm-wheels \
    --build-arg VLLM_REPO=https://github.com/local-inference-lab/vllm \
    --build-arg VLLM_REF=8244d1c44348ae8a5bbfc9f1b607021652283ed4 \
    --build-arg VLLM_APPLY_PRESET_PRS=0 --build-arg VLLM_PRESERVE_SM12X_TARGET=1 \
    --build-arg VLLM_PATCH_B12X_C128A_ALIGNMENT=1 .
test "$(cat artifacts/vllm-wheels/.vllm-commit)" = 8244d1c44348ae8a5bbfc9f1b607021652283ed4
test "$(cat artifacts/vllm-wheels/.vllm-arch)" = 12.1a

for arm in control candidate; do
    run_build --target flashinfer-export --output "type=local,dest=artifacts/flashinfer-$arm" \
        --build-arg FLASHINFER_REPO=https://github.com/local-inference-lab/flashinfer.git \
        --build-arg FLASHINFER_REF=91271757f7746f3dfb58f7c2aeb44a663389de9e \
        --build-arg FLASHINFER_BUILD_CUBIN=0 --build-arg FLASHINFER_APPLY_REGULAR_PATCHES=0 \
        --build-arg "DENSE_INT64_ARM=$arm" .
    test "$(cat "artifacts/flashinfer-$arm/.flashinfer-commit")" = 91271757f7746f3dfb58f7c2aeb44a663389de9e
    test "$(cat "artifacts/flashinfer-$arm/.flashinfer-arch")" = 12.1a
    run_build --target runner --tag "$tag_prefix-$arm" \
        --build-context "flashinfer_wheels=artifacts/flashinfer-$arm" \
        --build-context vllm_wheels=artifacts/vllm-wheels \
        --build-arg "DENSE_INT64_ARM=$arm" .
    docker image inspect "$tag_prefix-$arm" > "artifacts/image-$arm.json"
    docker run --rm --network none --entrypoint uv "$tag_prefix-$arm" \
        run --no-project --no-config --python /usr/bin/python3 -c \
        'import importlib.metadata as m,json; print(json.dumps(sorted((d.metadata["Name"].lower(),d.version) for d in m.distributions()),indent=2))' \
        > "artifacts/packages-$arm.json"
    docker run --rm --network none --entrypoint dpkg-query "$tag_prefix-$arm" \
        -W '-f=${Package}=${Version}\n' > "artifacts/debian-$arm.txt"
    docker run --rm --network none --entrypoint cat "$tag_prefix-$arm" \
        /workspace/dense-int64.json > "artifacts/treatment-$arm.json"
done
diff -u artifacts/packages-control.json artifacts/packages-candidate.json
diff -u artifacts/debian-control.txt artifacts/debian-candidate.txt
find artifacts -name '*.whl' -exec sha256sum {} + > artifacts/wheels-sha256.txt
printf '%s\n' 'Build pair complete. Native SM121 compilation, GPU correctness, sanitizer, graph replay and serving gates are still required.'
