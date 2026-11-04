#!/usr/bin/env bash
# Stop both GLM-5.3-Flash-FP4-Local ranks.
# Equivalent to: ./launch.sh down
exec "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/launch.sh" down "$@"
