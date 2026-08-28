#!/usr/bin/env bash
# download.sh — fetch GLM-5.3-Flash EXL3 (+ DFlash2) into the head HF cache
#
# Head only. Does not docker pull, SSH, or rsync to the worker. Weights land
# in $HF_HOME / ~/.cache/huggingface (~164 GiB target + ~2.3 GiB DFlash2).
# ./start.sh will rsync to the worker on launch unless SKIP_SYNC=1.
#
# Already complete? Exits 0. Re-fetch: REFRESH_WEIGHTS=1 ./download.sh
# Skip DFlash2: SPEC_METHOD=mtp ./download.sh
#
# Equivalent to: ./start.sh download
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/start.sh" download
