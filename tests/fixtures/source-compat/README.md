# Source-compat fixtures (tasks 27 + 13 + 17)

Compact synthetic copies of the live production contracts. They are not a
vLLM tree. `scripts/audit_source_compat.py` and the CPU tests consume them.

A full dump of the running image may exist at
`tests/fixtures/live-image-vllm/` for local audits. That dump is gitignored
and must not be committed.

Do not raise LPTT, force the V1 runner, or treat MTP as a configuration
rollback above 12 sequences.
