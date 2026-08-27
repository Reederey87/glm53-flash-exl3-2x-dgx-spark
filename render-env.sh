#!/usr/bin/env bash
# Render the node-local .env that docker compose reads.
#   render-env.sh <head|worker>
set -euo pipefail
ROLE="${1:?usage: render-env.sh <head|worker>}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT/cluster.env"

case "$ROLE" in
  head)   NODE_RANK=0; HEADLESS=""; VLLM_HOST_IP="$HEAD_R1" ;;
  worker) NODE_RANK=1; HEADLESS=1;  VLLM_HOST_IP="$WORKER_R1" ;;
  *) echo "unknown role $ROLE" >&2; exit 1 ;;
esac

# NOTE: unquoted heredoc -- '#' is literal and $VAR still expands. Keep any
# disabled line free of '$'.
cat > "$ROOT/.env" <<ENV
GLM53_IMAGE=$GLM53_IMAGE
GLM53_MODEL=$GLM53_MODEL
SERVED_MODEL_NAME=$SERVED_MODEL_NAME
HF_CACHE=$HF_CACHE
VLLM_HOST=$VLLM_HOST
API_PORT=$API_PORT
MAX_MODEL_LEN=$MAX_MODEL_LEN
MAX_NUM_SEQS=$MAX_NUM_SEQS
MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS
GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION
KV_CACHE_DTYPE=$KV_CACHE_DTYPE
BLOCK_SIZE=$BLOCK_SIZE
ATTENTION_BACKEND=$ATTENTION_BACKEND
MOE_BACKEND=$MOE_BACKEND
REASONING_EFFORT=$REASONING_EFFORT
MEM_MARGIN_GIB=$MEM_MARGIN_GIB
QSFP_IF=$QSFP_IF
NCCL_IB_HCA=$NCCL_IB_HCA
MASTER_ADDR=$MASTER_ADDR
MASTER_PORT=$MASTER_PORT
NODE_RANK=$NODE_RANK
HEADLESS=$HEADLESS
VLLM_HOST_IP=$VLLM_HOST_IP
ENV
chmod 600 "$ROOT/.env"
echo "rendered $ROOT/.env ($ROLE)"
