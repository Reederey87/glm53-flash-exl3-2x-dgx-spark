# Status

- 2026-10-03: Adopted the bounded resident-tail scheduler arm on both production ranks, retaining `glm53-selfbuild:e3-armc-shm50`; two same-shape races improved cached follow-up median by 3.68 s, all three five-run decode lanes cleared the pre-registered 97% floor, pool stayed at 567 blocks, and `/health` stayed 200. Receipts and gate details: `docs/24-resident-tail.md` and `local/resident-tail-20261003/` on spark1.

- 2026-09-29: Adopted token-exact host segment caching from `90cdc2d` on both production ranks after fresh exact-byte qualification, salt-isolation tests, independent approval and explicit adoption approval. Same image/KV geometry; receipts: `local/role-token-cache-20260929/v2/`, gates and rollback: `docs/24-role-token-cache.md`. Initial arm `73d8fb2` remains reverted.

- 2026-09-29: Adopted the metric-only sparse-retention miss overlay on both production ranks from commit `1dcb307`; exact-byte gate receipts are in `local/prefix-observability-20260928/` on spark1. See `docs/04-prefix-caching.md`.
