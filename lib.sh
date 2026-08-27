#!/usr/bin/env bash
# Shared ssh/rsync helpers. Source AFTER cluster.env.
# Deliberately small: node_ssh / node_rsync / node_label is the whole surface.

_ssh_opts=(-o ConnectTimeout="${SSH_CONNECT_TIMEOUT:-15}" -o BatchMode=yes)
[ -n "${SSH_KEY:-}" ] && _ssh_opts+=(-i "$SSH_KEY" -o IdentitiesOnly=yes)

# Resolve a name the way ssh would (macOS routes .local through mDNS here too).
_resolves() { python3 -c 'import socket,sys; socket.getaddrinfo(sys.argv[1],22)' "$1" >/dev/null 2>&1; }

_target() { # _target <head|worker> -> user@host, falling back to a pinned IP
  local host ip
  if [ "$1" = head ]; then host="$HEAD_HOST"; ip="${HEAD_IP:-}"; else host="$WORKER_HOST"; ip="${WORKER_IP:-}"; fi
  # An ssh_config alias is not a hostname; ask ssh what it expands to.
  local real; real="$(ssh -G "$host" 2>/dev/null | awk '/^hostname /{print $2; exit}')"
  [ -n "$real" ] || real="$host"
  if _resolves "$real" || [ -z "$ip" ]; then
    printf '%s@%s' "$CLUSTER_USER" "$host"
  else
    echo "WARN: '$real' did not resolve; falling back to $ip" >&2
    printf '%s@%s' "$CLUSTER_USER" "$ip"
  fi
}

node_label() { _target "$1"; }
# Callers pass a command string built from LOCAL variables and expect it
# expanded here, on the client, before it is sent. That is intentional.
# shellcheck disable=SC2029
node_ssh() { local r="$1"; shift; ssh "${_ssh_opts[@]}" "$(_target "$r")" "$@"; }
node_rsync() { # node_rsync <role> <src/> <dest/> [extra rsync flags...]
  local r="$1" src="$2" dest="$3"; shift 3
  # `-e "ssh ${_ssh_opts[*]}"` would flatten to a string rsync re-splits on
  # whitespace, breaking any SSH_KEY path containing a space -- which is the
  # normal case on macOS (~/Library/Application Support/...). Build the remote
  # shell with printf %q so each option survives as one word.
  local esc=""; local o
  for o in "${_ssh_opts[@]}"; do esc+=" $(printf '%q' "$o")"; done
  rsync -a "$@" -e "ssh$esc" "$src" "$(_target "$r"):$dest"
}
