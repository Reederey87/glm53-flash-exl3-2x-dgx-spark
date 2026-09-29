# Status

- 2026-09-29: Reverted token-exact segment cache arm `73d8fb2` after review found request salt arrived after tokenization. Corrected sync/async entrypoints are locally validated; requalification pending under `local/role-token-cache-20260929/v2/`. See `docs/24-role-token-cache.md`.

- 2026-09-29: Adopted the metric-only sparse-retention miss overlay on both production ranks from commit `1dcb307`; exact-byte gate receipts are in `local/prefix-observability-20260928/` on spark1. See `docs/04-prefix-caching.md`.
