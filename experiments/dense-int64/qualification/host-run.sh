#!/bin/bash
# Execute one native test arm on an already-idle node. Never starts serving.
set -euo pipefail
image=${1:?exact sha256 image ID}
arm=${2:?control or candidate}
receipts=${3:?new absolute receipt directory}
checks=$(cd -- "$(dirname -- "$0")" && pwd)
harness=/home/nvidia/glm53-native-tests/int64-20261103/harness
[[ $(id -un) == nvidia && $(uname -m) == aarch64 ]]
[[ "$image" =~ ^sha256:[a-f0-9]{64}$ ]]
[[ "$arm" == control || "$arm" == candidate ]]
[[ "$receipts" == /home/nvidia/glm53-native-tests/int64-20261103/results/* ]]
[[ ! -e "$receipts" ]]
[[ $(docker image inspect -f '{{.Id}}' "$image") == "$image" ]]
[[ -z $(docker ps -aq --filter name=vllm_node) ]]
[[ $(awk '/^MemFree:/ {print $2}' /proc/meminfo) -ge 94371840 ]]
[[ $(cat /proc/sys/vm/min_free_kbytes) == 1048576 ]]
[[ $(systemctl --user show glm53-build-dense-int64-20261103.service -p MainPID --value) == 0 ]]
if systemctl --user is-active --quiet vllm-glm53fp4.service ||
   systemctl --user is-active --quiet vllm-glm53fp4-watchdog.timer; then
    echo 'Serving service/watchdog must remain stopped' >&2
    exit 2
fi
(cd "$checks" && sha256sum --quiet -c sha256.txt)
printf '%s  %s\n' \
 fba66c81abc459ff1cec85ab833db229c888676c6c57f736d68deddfad42951d "$harness/pyproject.toml" \
 93789273aa70e920075c8a5ebaed28b38c9a5800ca1dbb8ecc555665914cf8fe "$harness/uv.lock" | sha256sum -c -
mkdir -p "$receipts"
exec > >(tee -a "$receipts/run.log") 2>&1
date -u +%FT%TZ > "$receipts/host-start.txt"
cat /proc/meminfo > "$receipts/meminfo-before.txt"
docker image inspect "$image" > "$receipts/image.json"
cp "$checks/sha256.txt" "$receipts/runner-sha256.txt"
set +e
docker run --rm --gpus all --cpus 4 --memory 64g --memory-swap 64g --shm-size 4g \
    --name "glm53-int64-native-$arm" \
    --mount "type=bind,src=$checks,dst=/checks,readonly" \
    --mount "type=bind,src=$harness,dst=/harness,readonly" \
    --mount "type=bind,src=$receipts,dst=/receipts" \
    --entrypoint bash "$image" /checks/container.sh "$arm"
status=$?
set -e
printf '%s\n' "$status" > "$receipts/docker-exit.txt"
cat /proc/meminfo > "$receipts/meminfo-after.txt"
date -u +%FT%TZ > "$receipts/host-end.txt"
exit "$status"
