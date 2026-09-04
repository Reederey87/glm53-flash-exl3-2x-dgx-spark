# W28 pinned-preview anchor snapshots

These minimal, syntax-valid snapshots preserve the exact patch anchors from
the immutable production image `glm53-selfbuild:b5ab8091-s2b` (image ID
`sha256:87a2cfd32aece0d9cf682fd3bc7821f9c074c7b3cb537017f8a1e9b96cdb80dc`):

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
