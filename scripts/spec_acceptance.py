#!/usr/bin/env python3
"""Measure MTP acceptance from the vLLM Prometheus endpoint.

A lifetime counter is not the run's rate. This script takes two scrapes
separated by ``--interval`` and reports the deltas:

    mean acceptance length = 1 + accepted / drafts
    draft acceptance rate   = accepted / drafted

where the counters are vLLM's
``vllm:spec_decode_num_{accepted_tokens,draft_tokens,drafts}_total``.

Usage:
  python3 scripts/spec_acceptance.py --base http://127.0.0.1:8000 --interval 30

Start the run, let it warm up, then measure a fixed workload. Standard library
only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

ACCEPTED = "vllm:spec_decode_num_accepted_tokens_total"
DRAFTED = "vllm:spec_decode_num_draft_tokens_total"
DRAFTS = "vllm:spec_decode_num_drafts_total"


@dataclass(frozen=True)
class Acceptance:
    accepted: float
    drafted: float
    drafts: float
    mean_acceptance_length: float
    draft_acceptance_rate: float
    interval_seconds: float
    resets: int


def parse_metrics(text: str) -> dict[str, float]:
    """Sum each counter across label sets."""
    counters: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            head, raw_value = line.rsplit(None, 1)
            value = float(raw_value)
        except ValueError:
            continue
        name = head.split("{", 1)[0]
        counters[name] = counters.get(name, 0.0) + value
    return counters


def _delta(before: dict[str, float], after: dict[str, float], name: str) -> tuple[float, int]:
    old = before.get(name, 0.0)
    new = after.get(name, 0.0)
    if new < old:  # counter reset (restart or rollover)
        return new, 1
    return new - old, 0


def compute(before: dict[str, float], after: dict[str, float], interval: float) -> Acceptance:
    accepted, reset_a = _delta(before, after, ACCEPTED)
    drafted, reset_d = _delta(before, after, DRAFTED)
    drafts, reset_s = _delta(before, after, DRAFTS)
    mean_length = 1.0 + (accepted / drafts) if drafts else 0.0
    rate = accepted / drafted if drafted else 0.0
    return Acceptance(
        accepted=accepted,
        drafted=drafted,
        drafts=drafts,
        mean_acceptance_length=mean_length,
        draft_acceptance_rate=rate,
        interval_seconds=interval,
        resets=reset_a + reset_d + reset_s,
    )


def scrape(base: str, api_key: str, timeout: float) -> str:
    url = f"{base.rstrip('/')}/metrics"
    headers = {"Accept": "text/plain"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="http://127.0.0.1:8000", help="server base URL")
    parser.add_argument("--interval", type=float, default=30.0, help="seconds between scrapes")
    parser.add_argument("--api-key", default="", help="optional bearer token")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout")
    parser.add_argument("--out", default="", help="write the JSON receipt here")
    args = parser.parse_args()

    try:
        before = parse_metrics(scrape(args.base, args.api_key, args.timeout))
        print(f"scraped; waiting {args.interval:.0f}s ...", file=sys.stderr)
        time.sleep(args.interval)
        after = parse_metrics(scrape(args.base, args.api_key, args.timeout))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"spec-acceptance: cannot scrape {args.base}/metrics: {exc}", file=sys.stderr)
        return 2

    missing = [name for name in (ACCEPTED, DRAFTED, DRAFTS) if name not in after]
    if missing:
        print(
            "spec-acceptance: metrics missing: " + ", ".join(missing) +
            ". Is speculative decoding enabled for this run?",
            file=sys.stderr,
        )
        return 2

    result = compute(before, after, args.interval)
    payload = {
        "base": args.base,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "acceptance": asdict(result),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if result.resets:
        print("spec-acceptance: counter reset detected; deltas use the post-reset value", file=sys.stderr)
    if result.drafts == 0:
        print("spec-acceptance: no draft steps in this window", file=sys.stderr)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
