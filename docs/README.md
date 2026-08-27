# Documentation

| Doc | What it covers |
|---|---|
| [01-architecture.md](01-architecture.md) | Why two Sparks, the memory model, QSFP wiring, one-time network setup |
| [02-parameters.md](02-parameters.md) | Every serving parameter and why it differs from the vendor recipe |
| [03-tp2-kernel-fix.md](03-tp2-kernel-fix.md) | **The core contribution**: the `(32, 2176)` sparse-MLA fix for TP=2 |
| [04-bringup.md](04-bringup.md) | Step-by-step deployment from a clean pair of nodes |
| [05-troubleshooting.md](05-troubleshooting.md) | Real failure signatures and what they mean |
| [06-what-does-not-work.md](06-what-does-not-work.md) | Dead ends, so you do not repeat them |

New here? Read 01, then 04. Read 03 if you want to know what this repo actually
adds over the upstream single-GPU work.
