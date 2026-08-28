#!/usr/bin/env bash
# BP-2b — GB10 hardware-fault (Xid) monitor. Long-running, continuous
# journalctl -k -f tail (NOT periodic polling — polling can miss a Xid line
# that lands between poll intervals or gets pushed out of the ring buffer by
# rotation under sustained load). Runs alongside watchdog.sh/metrics-watch.sh
# but is a categorically different signal: those are software-liveness checks
# that end in a service-pair BOUNCE; this one is a hardware-fault check that
# NEVER bounces anything, because a GPU that has gone off the PCIe bus (the
# documented DGX Spark GB10 Xid 119/154 GSP-firmware-timeout failure mode)
# cannot be recovered by restarting a systemd unit — cold power-cycle
# (30s+ unplug) is the only recovery (NVIDIA staff's own words on a real,
# live incident thread for this exact hardware class). Attempting a bounce
# here would just burn StartLimitBurst uselessly, a failure mode this project
# has already hit multiple times for unrelated reasons.
set -uo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
KIT="$(cd "$(dirname "$0")" && pwd)"
# LOCAL to this cluster: ported from 2xSPARK-CLUSTER/xid-monitor.sh for the
# EXL3 deployment. No cluster.env here; notify.env stays optional.
# shellcheck disable=SC1091
# shellcheck disable=SC1091
[ -f "$KIT/notify.env" ] && . "$KIT/notify.env" 2>/dev/null || true

INCIDENT_DIR="${INCIDENT_DIR:-$HOME/.local/state/glm53exl3-xid-incidents}"
mkdir -p "$INCIDENT_DIR" 2>/dev/null || true

# Catastrophic Xid classes — uncorrectable ECC / GPU-lost-off-bus / GSP-timeout /
# uncontained-error. These are the ones NVIDIA's own Xid catalog documents as
# needing a GPU reset or node reboot, and where the DGX Spark GB10 forum incident
# (Xid 119 + 154, GPU registers reading 0xbadf5600, full bus loss) showed software
# recovery does not work at all — only a cold power cycle does.
#   48  = ROBUST_CHANNEL_GPU_ECC_DBE (double-bit ECC error)
#   79  = GPU has fallen off the bus
#   94  = ROBUST_CHANNEL_CONTAINED_ERROR (app-scoped, but the containment class this
#         watches also covers the uncontained sibling below)
#   95  = ROBUST_CHANNEL_UNCONTAINED_ERROR (multi-app, GPU reset required)
#   119 = GSP firmware timeout (the exact class in the DGX Spark GB10 incident)
#   140 = UNRECOVERABLE_ECC_ERROR_ESCAPE
#   154 = (paired with 119 in the same incident — back-to-back timeout escalation)
CATASTROPHIC_XIDS=" 48 79 94 95 119 140 154 "

notify_send() {  # $1=text; silent no-op without creds, matches metrics-watch.sh
  [ -n "${TG_BOT_TOKEN:-}" ] && [ -n "${TG_CHAT_ID:-}" ] || return 1
  curl -fsS --max-time 8 -X POST \
    "https://api.telegram.org/bot${TG_BOT_TOKEN}/sendMessage" \
    --data-urlencode "chat_id=${TG_CHAT_ID}" \
    ${TG_THREAD_ID:+--data-urlencode "message_thread_id=${TG_THREAD_ID}"} \
    --data-urlencode "text=$1" \
    --data-urlencode "disable_web_page_preview=true" >/dev/null 2>&1
}

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

handle_xid() {  # $1 = xid code (digits only), $2 = the full matched log line
  local code="$1" line="$2" host stamp cur_log prev_log
  host="$(hostname -s 2>/dev/null || hostname 2>/dev/null || echo unknown-host)"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"

  if [[ "$CATASTROPHIC_XIDS" != *" $code "* ]]; then
    # Not in the catastrophic list — log-only, forensic completeness, no alert.
    echo "$(ts) xid-monitor: Xid $code seen (non-catastrophic per current list) — $line" >&2
    return 0
  fi

  cur_log="$INCIDENT_DIR/${stamp}_xid${code}_current-boot.log"
  prev_log="$INCIDENT_DIR/${stamp}_xid${code}_previous-boot.log"
  # NVIDIA's own recommended diagnostic capture, before anything else touches the box.
  journalctl -k -b >"$cur_log" 2>/dev/null || echo "$(ts) xid-monitor: journalctl -k -b capture failed" >&2
  journalctl -k -b -1 >"$prev_log" 2>/dev/null || true  # previous boot may not exist; non-fatal

  echo "$(ts) xid-monitor: CATASTROPHIC Xid $code on $host — $line" >&2
  echo "$(ts) xid-monitor: diagnostic logs saved to $cur_log (+ previous-boot if available)" >&2
  echo "$(ts) xid-monitor: NOT attempting any service bounce — hardware fault, software cannot recover this" >&2

  notify_send "XID-FAULT (hardware) on ${host}: Xid ${code} detected — GPU may be off the PCIe bus. Software restart will NOT recover this. Physical power-cycle (30s+ unplug) required. Diagnostic logs saved to ${cur_log}." \
    || echo "$(ts) xid-monitor: Telegram alert not sent (no notify.env creds, or transport failure) — see log lines above" >&2
}

process_line() {  # $1 = one journalctl -k line; isolated for testability (see --test below)
  local line="$1" code
  case "$line" in
    *Xid*)
      # Typical NVRM format: "NVRM: Xid (PCI:0000:0f:00): 119, pid=..., ...".
      # The naive "first digit run after the word Xid" is WRONG — it matches
      # the PCI bus address (e.g. "0000") inside the parenthetical, not the
      # actual code that follows it. Strip a leading "(...)" device-id
      # parenthetical first (if present — some formats omit it), THEN take
      # the first digit run; this correctly extracts 119, not 0000, from the
      # example above. Verified against both the PCI-parenthetical and the
      # bare "Xid: N," forms via --test.
      after_xid="${line#*Xid}"
      stripped="$(printf '%s' "$after_xid" | sed -E 's/^[^0-9(]*\([^)]*\)//')"
      code="$(printf '%s\n' "$stripped" | grep -oE '[0-9]+' | head -1)"
      if [ -n "$code" ]; then
        handle_xid "$code" "$line"
      else
        echo "$(ts) xid-monitor: line matched 'Xid' but no code could be parsed — $line" >&2
      fi
      ;;
  esac
}

# --test mode: exercise the REACTION path directly (handle_xid/process_line)
# without touching the real kernel ring buffer. This matters because genuine
# Xid lines arrive via _TRANSPORT=kernel (journalctl -k / --dmesg), but a
# synthetic injection via userspace `logger` lands via _TRANSPORT=syslog
# instead — `journalctl -k` would NOT see it, and the `nvidia` cluster user
# has no sudo to write /dev/kmsg directly (AGENTS.md: docker group only, no
# sudo). So the gate test calls this script with a fake line argument and
# verifies the alert/capture/no-bounce behavior in isolation — see
# docs/xid-monitor-README.md for the exact invocation.
if [ "${1:-}" = "--test" ]; then
  shift
  test_line="${1:-NVRM: Xid (PCI:0000:0f:00): 119, pid=1234, name=test, synthetic test injection}"
  echo "$(ts) xid-monitor: --test mode, injecting synthetic line: $test_line" >&2
  process_line "$test_line"
  exit 0
fi

echo "$(ts) xid-monitor: starting continuous journalctl -k -f tail" >&2

# Continuous tail, not polling. `stdbuf -oL` keeps journalctl's output
# line-buffered through the pipe so matches are handled as soon as they land,
# not batched behind pipe buffering.
stdbuf -oL journalctl -k -f -o cat 2>/dev/null | while IFS= read -r line; do
  process_line "$line"
done

echo "$(ts) xid-monitor: journalctl -k -f pipe ended — exiting (systemd Restart=always will relaunch)" >&2
