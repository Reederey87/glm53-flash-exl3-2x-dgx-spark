# Shared-expert anchor snapshot — fork `487ecf187`

`shared_experts.py` is a byte-exact copy of the file inside the immutable
production image `glm53-selfbuild:e3-armc-guards`:

    /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/
        fused_moe/runner/shared_experts.py

- vLLM in that image: `0.1.dev20051+g487ecf187`
- sha256 of this snapshot: `13ec6272802ceeca754c4907c738eed23afc093c2b7b93658983a211e51c754d`
- 184 lines

Unlike the minimal W28 anchor snapshots, this file is complete and executable:
`tests/test_shared_experts_overlap.py` loads it through a stubbed `torch` and
`vllm` and drives `SharedExperts` directly, so the test proves the enqueue
*ordering* rather than only the anchor text. A snapshot copied from a different
vLLM revision fails the anchor preflight, which is the intent.
