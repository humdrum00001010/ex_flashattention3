#!/usr/bin/env bash
set -euo pipefail

mkdir -p /root/.ssh /run/sshd
chmod 700 /root/.ssh

if [[ -n "${PUBLIC_KEY:-}" ]]; then
  printf '%s\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
fi

ssh-keygen -A -q
/usr/sbin/sshd

exec "$@"
