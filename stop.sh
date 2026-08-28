#!/usr/bin/env bash
# stop.sh — stop the GLM-5.3-Flash EXL3 vLLM server started by start.sh
#
# Removes glm53-exl3-head on this machine and glm53-exl3-worker on the
# worker (same OS user as the head unless .env sets WORKER_USER). Weights
# and compile caches stay on disk so a later ./start.sh restarts fast.
#
# Equivalent to: ./start.sh stop
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/start.sh" stop
