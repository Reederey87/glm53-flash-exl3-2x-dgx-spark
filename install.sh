#!/usr/bin/env bash
# Push the kit to both nodes and install the systemd user units.
# Units are installed but NOT enabled -- enable them only when you want the
# cluster to serve on boot (see docs/04-bringup.md).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
# Check BEFORE sourcing: under `set -e` a missing file aborts at the source and
# this message -- the only guidance for the commonest first-run mistake -- would
# never print.
[ -f "$ROOT/cluster.env" ] || {
  echo "FAIL: no cluster.env. Run:  cp cluster.env.example cluster.env  and edit it." >&2
  exit 1
}
# shellcheck disable=SC1091
source "$ROOT/cluster.env"; source "$ROOT/lib.sh"

install_node() {
  local role="$1" unit="$2" host; host="$(node_label "$role")"
  node_ssh "$role" "mkdir -p '$KIT_DIR'"
  node_rsync "$role" "$ROOT/" "$KIT_DIR/" \
    --exclude='.env' --exclude='.git/' --exclude='*.log' --exclude='__pycache__' \
    || { echo "FAIL: rsync to $host" >&2; exit 1; }
  node_ssh "$role" "chmod +x '$KIT_DIR'/*.sh '$KIT_DIR'/bench/*.sh '$KIT_DIR'/image/*.sh 2>/dev/null || true"
  # Units ship with @KIT_DIR@ placeholders so KIT_DIR stays configurable.
  node_ssh "$role" "mkdir -p ~/.config/systemd/user && sed 's|@KIT_DIR@|$KIT_DIR|g' '$KIT_DIR/systemd/$unit' > ~/.config/systemd/user/$unit && systemctl --user daemon-reload" \
    || { echo "FAIL: unit install on $host" >&2; exit 1; }
  echo "ok: $host <- kit + $unit"
}

install_node head   vllm-glm53-head.service
install_node worker vllm-glm53-worker.service

# Watchdog lives on the head only (it probes the API and orders the pair bounce).
for u in vllm-glm53-watchdog.service vllm-glm53-watchdog.timer; do
  # $u must expand HERE, not on the node.
  node_ssh head "sed 's|@KIT_DIR@|$KIT_DIR|g' '$KIT_DIR/systemd/$u' > ~/.config/systemd/user/$u"
done
node_ssh head 'systemctl --user daemon-reload'
echo "ok: watchdog units installed on head"

# systemd user units only survive logout with lingering enabled.
for role in head worker; do
  L="$(node_ssh "$role" "loginctl show-user $CLUSTER_USER --property=Linger --value" 2>/dev/null || echo unknown)"
  [ "$L" = yes ] || echo "WARN: linger is '$L' on $(node_label "$role") -- run: sudo loginctl enable-linger $CLUSTER_USER" >&2
done
echo "ok: installed. Next: ./start-cluster.sh"
