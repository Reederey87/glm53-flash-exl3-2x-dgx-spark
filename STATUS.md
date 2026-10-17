# Status

- 2026-10-17: Replaced the EXL3 kit with `nvidia/GLM-5.3-Flash-FP4-Local` on
  `eugr/spark-vllm-b12x:latest`, driven by the pinned `eugr/spark-vllm-docker`
  one-step launcher. The recipe targets a 2M-token FP8 KV pool
  (`--kv-cache-memory-bytes 11G`), MTP-5, 500k context, TP2. All EXL3 code,
  Dockerfiles, overlays, and EXL3 receipts were removed from the tree; the old
  kit remains in git history.
- Cluster deployment of the new recipe is pending a maintenance window. Do not
  stop the running EXL3 production until the FP4 image, checkpoint, KV gate,
  and smoke receipt are captured on the pair.
