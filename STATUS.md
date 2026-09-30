# Status

- 2026-09-29: Adopted token-exact host segment caching from `90cdc2d` on both production ranks after fresh exact-byte qualification, salt-isolation tests, independent approval and explicit adoption approval. Same image/KV geometry; receipts: `local/role-token-cache-20260929/v2/`, gates and rollback: `docs/24-role-token-cache.md`. Initial arm `73d8fb2` remains reverted.

- 2026-09-29: Adopted the metric-only sparse-retention miss overlay on both production ranks from commit `1dcb307`; exact-byte gate receipts are in `local/prefix-observability-20260928/` on spark1. See `docs/04-prefix-caching.md`.
