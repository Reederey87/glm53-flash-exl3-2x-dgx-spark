#!/usr/bin/env bash
# Prepare (or revert) GPU performance-counter access on a DGX Spark node.
#
#   sudo bash scripts/enable-gpu-profiling.sh          # write the drop-in
#   sudo bash scripts/enable-gpu-profiling.sh --check  # report only
#   sudo bash scripts/enable-gpu-profiling.sh --revert # remove it
#
# The change is one module option, NVreg_RestrictProfilingToAdminUsers=0, in a
# dedicated drop-in. It is read when the nvidia module loads, so a reboot is
# required and this script NEVER reboots on its own. See
# docs/14-profiling-privilege-runbook.md for the full procedure and rollback.
#
# It refuses to touch a drop-in it does not own (identified by a managed-by
# marker); use --force only after inspecting the file. GLM53_PROFILING_DROPIN
# redirects the target path for tests; the real path always needs root.
set -euo pipefail

SYSTEM_DROPIN=/etc/modprobe.d/99-nvidia-profiling.conf
DROPIN="${GLM53_PROFILING_DROPIN:-$SYSTEM_DROPIN}"
PARAMS=/proc/driver/nvidia/params
OPTION='options nvidia NVreg_RestrictProfilingToAdminUsers=0'
MARKER='managed by glm53 enable-gpu-profiling.sh'

say() { printf '%s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 2; }

is_system() { [ "$DROPIN" = "$SYSTEM_DROPIN" ]; }

require_root() {
    if is_system && [ "$(id -u)" -ne 0 ]; then
        die "run this with sudo (it writes $DROPIN)"
    fi
}

is_managed() {
    [ -f "$DROPIN" ] && grep -qF "$MARKER" "$DROPIN"
}

show_state() {
    local live="unknown"
    if [ -r "$PARAMS" ]; then
        live="$(awk -F': *' '/RmProfilingAdminOnly/ {print $2}' "$PARAMS")"
        say "live driver  RmProfilingAdminOnly=${live:-unknown}"
    else
        say "live driver  $PARAMS not readable"
    fi
    if [ -f "$DROPIN" ]; then
        if is_managed; then
            say "drop-in      $DROPIN present and managed:"
        else
            say "drop-in      $DROPIN present but NOT managed by this script:"
        fi
        sed 's/^/  /' "$DROPIN"
    else
        say "drop-in      $DROPIN absent"
    fi
    if [ "$live" = "0" ]; then
        say "state        counters enabled"
    elif [ "$live" = "1" ]; then
        say "state        counters root-only (default); reboot after writing the drop-in"
    fi
    if command -v nsys >/dev/null 2>&1; then
        say "nsys         $(nsys --version 2>/dev/null | head -1)"
    else
        say "nsys         not found"
    fi
    if command -v ncu >/dev/null 2>&1; then
        say "ncu          $(ncu --version 2>/dev/null | awk '/Version/ {print $NF; exit}')"
    else
        say "ncu          not installed (nsys --gpu-metrics is enough for byte counters)"
    fi
}

report_conflicts() {
    local hits
    hits="$(grep -rls 'NVreg_RestrictProfilingToAdminUsers' /etc/modprobe.d 2>/dev/null || true)"
    if [ -n "$hits" ]; then
        say "existing references:"
        printf '%s\n' "$hits" | sed 's/^/  /'
    fi
}

rebuild_initramfs() {
    if ! is_system; then
        say "skip initramfs rebuild (non-system drop-in: $DROPIN)"
        return 0
    fi
    if command -v update-initramfs >/dev/null 2>&1; then
        update-initramfs -u -k all
        say "initramfs rebuilt"
    else
        say "note: update-initramfs not found; rebuild your initramfs if your distro needs it"
    fi
}

apply() {
    require_root
    if ! awk '{print $1}' /proc/modules 2>/dev/null | grep -qx nvidia; then
        say "note: the nvidia module is not loaded right now"
    fi
    if [ -f "$DROPIN" ] && ! is_managed; then
        if [ "$FORCE" -ne 1 ]; then
            die "$DROPIN exists and is not managed by this script; inspect it and use --force to replace it (a backup is then kept)"
        fi
        cp -p "$DROPIN" "$DROPIN.bak-$(date +%Y%m%d-%H%M%S)"
        say "backed up the existing drop-in"
    fi
    if is_managed && grep -qF "$OPTION" "$DROPIN"; then
        say "already present: $OPTION"
    else
        printf '# Enable GPU performance counters for local users (docs/14).\n# %s\n# Revert: sudo bash scripts/enable-gpu-profiling.sh --revert\n%s\n' \
            "$MARKER" "$OPTION" > "$DROPIN"
        say "wrote $DROPIN"
    fi
    rebuild_initramfs
    report_conflicts
    show_state
    say
    say "Next: reboot this node (the option is read at module load), then run"
    say "  nsys profile --gpu-metrics-devices=help   # expect NVIDIA GB10, no privilege error"
}

revert() {
    require_root
    if [ ! -f "$DROPIN" ]; then
        say "nothing to remove"
    elif ! is_managed; then
        if [ "$FORCE" -ne 1 ]; then
            die "$DROPIN is not managed by this script; refusing to delete it (use --force only if you are sure)"
        fi
        rm -f "$DROPIN"
        say "removed unmanaged $DROPIN (--force)"
    else
        rm -f "$DROPIN"
        say "removed $DROPIN"
    fi
    rebuild_initramfs
    show_state
    say
    say "Next: reboot this node to return the profiling APIs to root-only."
}

usage() {
    say "usage: sudo bash scripts/enable-gpu-profiling.sh [--check|--revert|--force]"
    say "  (no argument)  write the module drop-in and rebuild the initramfs"
    say "  --check        report the live driver flag, drop-in and tool state"
    say "  --revert       remove the drop-in and rebuild the initramfs"
    say "  --force        replace/delete a drop-in this script does not own"
    say "See docs/14-profiling-privilege-runbook.md. This script never reboots."
}

MODE=apply
FORCE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --apply)  MODE=apply ;;
        --check)  MODE=check ;;
        --revert) MODE=revert ;;
        --force)  FORCE=1 ;;
        -h|--help) MODE=help ;;
        *) die "unknown argument: $1 (use --check, --revert, --force, or no argument)" ;;
    esac
    shift
done

case "$MODE" in
    apply)  apply ;;
    check)  show_state ;;
    revert) revert ;;
    help)   usage ;;
esac
