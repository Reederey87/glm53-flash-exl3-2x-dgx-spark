#!/usr/bin/env bash
# Pull the serving image and download/copy the checkpoint without starting it.
# Equivalent to: ./launch.sh setup
exec "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/launch.sh" setup "$@"
