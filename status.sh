#!/usr/bin/env bash
# Show the GLM-5.3-Flash-FP4-Local cluster and API health.
# Equivalent to: ./launch.sh status
exec "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/launch.sh" status "$@"
