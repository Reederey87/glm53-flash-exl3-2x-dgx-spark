# Manual recovery of accumulated watermark boosts

This is an experiment-window recovery operation, separate from candidate26's
service readiness patch. It is not installed into the launcher or service.
It changes no configured reserve, scaling factor, boost factor or allocator.

After the failed fresh-incumbent startup, read-only `/proc/zoneinfo` inspection
found head Normal-zone boost **586588 pages** versus worker **7680 pages**
(4096-byte pages). Both still have `min_free_kbytes=1048576`, scale10 and
boost factor15000. Head base min/low/high are260707/325883/391059 pages;
reported watermarks include the extra boost. This explains an available-memory
penalty beyond the configured reserve; ordinary file-page advice does not
necessarily remove that penalty. It does not prove that file cache is the only
remaining cause of the startup failure.

## Research and stronger alternatives

Exa lookup covered [Linux6.17 page allocation source](https://github.com/torvalds/linux/blob/v6.17/mm/page_alloc.c),
[MemAvailable computation](https://github.com/torvalds/linux/blob/master/mm/show_mem.c),
[VM sysctl documentation](https://docs.kernel.org/admin-guide/sysctl/vm.html),
and the [historical reset-order fix](https://lists.openwall.net/linux-kernel/2020/05/14/574).
In6.17, a successful write to `min_free_kbytes` calls `setup_per_zone_wmarks`
even when its numeric value is unchanged. The setup resets each zone's boost
before deriving low/high watermarks and recalculates reserves. This is source
evidence for upstream6.17; verify the actual vendor kernel's result on each node.

Disabling boosting, lowering the reserve or utilization, or introducing an
allocator patch would change policy and are excluded from this baseline.
A reboot is more disruptive. Global cache eviction is unnecessary for testing
this specific diagnosis and is not part of the operation. Reapplying the same
value clears transient state while retaining the configured policy.

Local-tree review followed Exa: vLLM uses host available memory on integrated
GPUs and separately checks utilization; Exllamav3 preserves host headroom when
computing an integrated-device budget; TensorFold releases retired snapshot
references before allocator cleanup. None justifies bypassing a startup check
or changing this checkpoint/KV pool. No engine code is modified.

## Frozen operation and gate

Freeze `refresh_watermarks.py` and its fault tests before staging. Snapshot
zoneinfo, meminfo, kernel identity and the three numeric sysctls on BOTH nodes.
Require no containers/GPU work/builds and an inactive watchdog. Preflight the
helper on both nodes with the reserve file mounted read-only and no `--apply`.

The apply container has no network, a read-only root, all capabilities dropped,
`no-new-privileges`, a64MiB temporary directory and256MiB memory limit. Its
**only writable host mount** is `/proc/sys/vm/min_free_kbytes`, mounted as
`/host-min-free-kbytes`. No privileged mode, broad proc mount, Docker socket,
host root, model cache or security-profile override is required. If the default
security profile refuses the write, preserve the refusal and reassess rather
than broadening the container privileges.

The helper requires the qualified ARM64 kernel6.17.0-1031-nvidia and exact
existing values1048576/10/15000. It writes only `1048576\n`, then requires
unchanged numeric settings, managed pages, zone set and base min/low/high
watermarks, with boost0. Reapply on BOTH nodes; a failure does not authorize
a different value. No persistent configuration file is edited. The configured
rollback value remains1048576; transient boosts can naturally accumulate again.

Record actual memory and watermarks before/after. This operation does not claim
to reclaim file pages or to pass the90GiB MemFree gate. If that gate remains
unsatisfied, recover only separately verified completed-file caches. Before
every serving arm, retain the same numeric settings and check current watermarks;
use this same recovery only if needed and record it. Verify BOTH-node readiness,
actual startup health and exact capacity before any benchmark. All original
functional and full speed gates remain mandatory.

Fifteen local tests cover exact mount/mode scope, refusing altered reserve,
scale/factor/base watermarks or managed pages, incomplete reset, malformed zone
data and duplicate fields. The read-only real-node preflight and actual reset
receipts remain required; these unit tests cannot prove kernel behavior.
