# W28 pinned-preview anchor snapshots

These minimal, syntax-valid snapshots preserve the exact patch anchors from
the deployed GLM-5.3 preview source at vLLM commit `933876c388`:

- `vllm/v1/attention/backends/mla/indexer.py`
- `vllm/models/glm5next/nvidia/attention.py`
- `vllm/v1/worker/gpu/model_states/interface.py`
- `vllm/v1/worker/gpu/model_runner.py`
- `vllm/v1/worker/gpu/model_states/mamba_hybrid.py`
- `vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py`
- `vllm/model_executor/layers/sparse_attn_indexer_kpool.py`

They are deliberately independent of the overlay constants. Tests apply the
overlays to these files by default, so an anchor copied from a different vLLM
revision fails in CI. Environment overrides still allow the same tests to run
against complete files copied from a container.
