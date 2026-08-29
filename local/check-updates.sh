#!/usr/bin/env bash
# LOCAL to this cluster; not part of the vendored upstream kit.
#
# Weekly update + parity check for the GLM-5.3-Flash-EXL3 pair. READ-ONLY:
# safe while the model serves. Adapted from glm53-flash-nvfp4/check-updates.sh.
#
#   bash check-updates.sh          # report + exit non-zero if action is needed
#   bash check-updates.sh --quiet  # only print when something needs attention
#
# Four questions:
#   1. Have spark1/spark2 drifted? One TP=2 model; asymmetry has taken this
#      cluster down before (vm.min_free_kbytes, nvidia-fs on one node only).
#   2. Is a newer DGX OTA offered? The only vendor upgrade signal worth acting
#      on (Ubuntu's generic driver branches are vendor-pinned away).
#   3. Has GHCR's :exl3 tag moved off our pinned digest? Drift = upstream
#      rebuilt the image; production is unaffected (we pin by digest) but a
#      review is due.
#   4. Has the upstream recipe repo moved past our staged commit? New commits
#      = candidate fixes to review (this is how we caught the JIT-cache and
#      stop-strings fixes on 2026-08-28).
#
# Notification design: the systemd service has NO SuccessExitStatus override,
# so exit 1 leaves the unit failed and visible in `systemctl --user --failed`.
set -uo pipefail
KIT="$(cd "$(dirname "$0")/.." && pwd)"
QUIET=0; [ "${1:-}" = "--quiet" ] && QUIET=1

WORKER_SSH="${WORKER_SSH:-nvidia@192.168.177.11}"
QSFP_IF="${QSFP_IF:-enp1s0f1np1}"
UPSTREAM_REPO="${UPSTREAM_REPO:-https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks}"
GHCR_REPO="miaai-lab/glm-5.3-flash-2x-dgx-sparks"

say()  { [ "$QUIET" -eq 1 ] || printf '%s\n' "$*"; }
warn() { printf '%s\n' "$*"; }   # always printed

PROBE='
echo "ota=$(grep -oP "DGX_OTA_VERSION=\"\K[^\"]+" /etc/dgx-release 2>/dev/null)"
echo "ota_candidate=$(apt-cache policy dgx-release 2>/dev/null | awk "/Candidate:/{print \$2}")"
echo "kernel_running=$(uname -r)"
echo "kernel_installed=$(dpkg-query -W -f=\${Version} linux-image-nvidia-hwe-24.04 2>/dev/null)"
echo "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
echo "cuda=$(nvidia-smi 2>/dev/null | grep -oE "CUDA Version: [0-9.]+" | grep -oE "[0-9.]+$")"
echo "vbios=$(nvidia-smi --query-gpu=vbios_version --format=csv,noheader 2>/dev/null | head -1)"
echo "nic_fw=$(ethtool -i '"$QSFP_IF"' 2>/dev/null | awk "/firmware-version/{print \$2}")"
echo "nvidia_fs=$(dpkg-query -W -f=\${Version} linux-modules-nvidia-fs-nvidia-hwe-24.04 2>/dev/null || echo ABSENT)"
echo "container_toolkit=$(dpkg-query -W -f=\${Version} nvidia-container-toolkit 2>/dev/null)"
echo "docker=$(docker --version 2>/dev/null | grep -oE "[0-9]+\.[0-9]+\.[0-9]+" | head -1)"
echo "min_free_kbytes=$(cat /proc/sys/vm/min_free_kbytes 2>/dev/null)"
echo "upgradable=$(apt list --upgradable 2>/dev/null | tail -n +2 | wc -l)"
echo "reboot_required=$([ -f /var/run/reboot-required ] && echo yes || echo no)"
echo "apt_cache_age_days=$(python3 -c "
import os,time
p=\"/var/lib/apt/periodic/update-success-stamp\"
try: print(int((time.time()-os.path.getmtime(p))//86400))
except Exception: print(-1)
")"
'

tmp_h="$(mktemp "${TMPDIR:-/tmp}/glm53exl3upd.XXXXXX")"
tmp_w="$(mktemp "${TMPDIR:-/tmp}/glm53exl3upd.XXXXXX")"
trap 'rm -f "$tmp_h" "$tmp_w"' EXIT

# GLM53_PROBE_{HEAD,WORKER}: pre-captured key=value snapshots for offline
# testing of the mismatch paths without touching a production node.
if [ -n "${GLM53_PROBE_HEAD:-}" ] && [ -n "${GLM53_PROBE_WORKER:-}" ]; then
  cp "$GLM53_PROBE_HEAD" "$tmp_h"; cp "$GLM53_PROBE_WORKER" "$tmp_w"
else
  bash -c "$PROBE" 2>/dev/null | grep -E '^[a-z_]+=' > "$tmp_h"
  ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
      "$WORKER_SSH" "$PROBE" 2>/dev/null | grep -E '^[a-z_]+=' > "$tmp_w"
fi

get() { awk -F= -v k="$2" '$1==k{sub(/^[^=]*=/,""); print}' "$1"; }

LBL_H="$(hostname -s 2>/dev/null || echo head)"; LBL_W="$WORKER_SSH"
[ -s "$tmp_h" ] || { warn "FAIL: could not probe $LBL_H"; exit 2; }
[ -s "$tmp_w" ] || { warn "FAIL: could not probe $LBL_W"; exit 2; }

problems=0

say "== EXL3 cluster update / parity check  ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
say ""
say "$(printf '%-20s %-26s %s' 'field' "$LBL_H" "$LBL_W")"
say "$(printf '%-20s %-26s %s' '-----' '----' '----')"

PARITY="ota kernel_running kernel_installed driver cuda vbios nic_fw nvidia_fs container_toolkit docker min_free_kbytes"
for k in $PARITY reboot_required upgradable apt_cache_age_days; do
  a="$(get "$tmp_h" "$k")"; b="$(get "$tmp_w" "$k")"
  mark=""
  case " $PARITY " in
    *" $k "*) [ "$a" != "$b" ] && { mark="   <== MISMATCH"; problems=$((problems+1)); } ;;
  esac
  # Mismatches always print (even --quiet) so the journal carries the detail,
  # not just the failed exit.
  if [ -n "$mark" ]; then
    warn "$(printf '%-20s %-26s %s%s' "$k" "${a:-?}" "${b:-?}" "$mark")"
  else
    say "$(printf '%-20s %-26s %s%s' "$k" "${a:-?}" "${b:-?}" "$mark")"
  fi
done
say ""

# --- 1. DGX OTA -----------------------------------------------------------
ota_h="$(get "$tmp_h" ota)"; cand_h="$(get "$tmp_h" ota_candidate)"
age="$(get "$tmp_h" apt_cache_age_days)"
if [ "$age" = "-1" ]; then
  warn "NOTE: apt cache age unknown on the head — 'no new OTA' may be stale."
elif [ "${age:-0}" -gt 7 ] 2>/dev/null; then
  warn "NOTE: apt cache is ${age} days old. Run 'sudo apt-get update' before"
  warn "      trusting the OTA candidate below."
fi
if [ -n "$cand_h" ] && [ "$cand_h" != "$ota_h" ] && [ "$cand_h" != "(none)" ]; then
  warn ""
  warn "*** DGX OTA UPDATE AVAILABLE: $ota_h -> $cand_h ***"
  warn "    Plan it: stop the model, upgrade BOTH nodes, reboot both, then re-run"
  warn "    local/acceptance.sh and local/serving-test.sh."
  warn "    ⚠ Driver hold: 590.x has a GB10 CUDAGraph deadlock (field reports,"
  warn "    2026-08). Before applying, confirm the OTA does NOT carry a 590.x"
  warn "    driver; stay on the 580.x branch until that is fixed."
  problems=$((problems+1))
else
  say "DGX OTA: ${ota_h:-?} — current (no newer release offered)."
fi

# --- 1b. driver branch hold (590.x CUDAGraph deadlock on GB10) ------------
for n in h w; do
  eval "drv=\"\$(get \"\$tmp_$n\" driver)\""
  case "$drv" in
    590.*)
      warn ""
      warn "*** DRIVER $drv IS ON THE 590.x BRANCH ($( [ "$n" = h ] && echo head || echo worker )) ***"
      warn "    590.x deadlocks CUDA graph capture on GB10 — prod runs graphs ON."
      warn "    Roll back to the 580.x branch."
      problems=$((problems+1));;
  esac
done

# --- 2. reboot pending ----------------------------------------------------
if [ "$(get "$tmp_h" reboot_required)" = "yes" ] || [ "$(get "$tmp_w" reboot_required)" = "yes" ]; then
  warn ""
  warn "*** REBOOT PENDING — a kernel/driver update is staged but NOT running. ***"
  warn "    Until both nodes reboot they run the old kernel. Reboot BOTH."
  problems=$((problems+1))
fi

# --- 3. GHCR :exl3 tag drift vs our digest pin ----------------------------
pinned="$(grep -oE 'sha256:[0-9a-f]{64}' "$KIT/.env" 2>/dev/null | head -1)"
if [ -n "$pinned" ]; then
  tok="$(curl -fsS --max-time 20 "https://ghcr.io/token?scope=repository:${GHCR_REPO}:pull" 2>/dev/null \
        | grep -oE '"token":"[^"]+"' | cut -d'"' -f4)"
  live="$(curl -fsS --max-time 20 -H "Authorization: Bearer $tok" \
        -H "Accept: application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.v2+json, application/vnd.docker.distribution.manifest.list.v2+json" \
        -o /dev/null -D - "https://ghcr.io/v2/${GHCR_REPO}/manifests/exl3" 2>/dev/null \
        | grep -i docker-content-digest | grep -oE 'sha256:[0-9a-f]{64}')"
  if [ -z "$live" ]; then
    warn "NOTE: could not query GHCR for the :exl3 tag (offline?) — image drift unchecked."
  elif [ "$live" != "$pinned" ]; then
    warn ""
    warn "*** GHCR :exl3 TAG MOVED: now $live (pinned $pinned). ***"
    warn "    Production is unaffected (digest pin), but upstream rebuilt the image —"
    warn "    review their changelog before considering an update."
    problems=$((problems+1))
  else
    say "GHCR :exl3 digest: matches our pin (${pinned:7:12}…)."
  fi
else
  warn "NOTE: no sha256 digest pin found in $KIT/.env — cannot check image drift."
fi

# --- 4. upstream recipe repo drift ----------------------------------------
staged="$(cat "$KIT/.staged-upstream-commit" 2>/dev/null)"
if [ -n "$staged" ]; then
  remote="$(git ls-remote "$UPSTREAM_REPO" HEAD 2>/dev/null | awk '{print $1}')"
  if [ -z "$remote" ]; then
    warn "NOTE: could not reach $UPSTREAM_REPO (offline?) — recipe drift unchecked."
  elif [ "${remote:0:7}" != "${staged:0:7}" ]; then
    warn ""
    warn "*** UPSTREAM RECIPE MOVED: HEAD ${remote:0:7} vs staged ${staged:0:7}. ***"
    warn "    New commits are review candidates (2026-08-28 precedent: JIT-cache"
    warn "    persistence + stop-strings fix). Review on the Mac copy, then sync."
    problems=$((problems+1))
  else
    say "Upstream recipe: staged ${staged:0:7} is current."
  fi
else
  warn "NOTE: $KIT/.staged-upstream-commit missing — recipe drift unchecked."
fi

say ""
if [ "$problems" -eq 0 ]; then
  say "RESULT: OK — nodes in parity, no vendor update, image and recipe current."
  exit 0
fi
warn ""
warn "RESULT: $problems item(s) need attention (see above)."
exit 1
