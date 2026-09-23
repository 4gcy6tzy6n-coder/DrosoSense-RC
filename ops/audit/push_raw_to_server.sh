#!/bin/bash
# Robust per-file upload for the FlyWire raw inputs.
#
# The link goes through a local proxy that resets long-lived connections
# (observed: rsync "unexpected end of file" / "Connection closed by
# 198.18.0.13"), so a single 9.5 GB stream does not survive. This loop
# transfers ONE file per rsync invocation with --append, and retries the current
# file until it completes. --append-verify is NOT used because the local rsync is
# openrsync, which does not implement it. The weaker append is acceptable here
# only because the whole file is sha256-verified against the committed manifest
# afterwards, which is what actually establishes the copy is correct.
#
# Usage: ops/audit/push_raw_to_server.sh [dest_dir]
set -u

SRC="/Users/yyl/Desktop/workshop/DrosoSense-RC/data-root/connectome/raw"
DEST="${1:-/root/autodl-tmp/drososense/data-root/connectome/raw}"
HOST="root@connect.nmb2.seetacloud.com"
PORT=31651
KEY="/Users/yyl/.ssh/id_ed25519"
SSH_OPTS="-p $PORT -i $KEY -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=20 -o ServerAliveCountMax=4"
ATTEMPTS=12

FILES=(
  "flywire_synapses_783 (1).feather"
  "proofread_connections_783.feather"
  "per_neuron_neuropil_count_post_783.feather"
  "per_neuron_neuropil_count_pre_783.feather"
  "neuron_class_ranking_df_783-olfactory-10000.feather"
  "proofread_root_ids_783.npy"
)

fail=0
for f in "${FILES[@]}"; do
  local_size=$(stat -f %z "$SRC/$f" 2>/dev/null || echo 0)
  echo "=== $f  (local ${local_size} bytes) ==="
  ok=0
  for attempt in $(seq 1 "$ATTEMPTS"); do
    rsync -a --append --partial --timeout=120 -e "ssh $SSH_OPTS" \
      "$SRC/$f" "$HOST:$DEST/" && { ok=1; break; }
    echo "    attempt $attempt failed; resuming in 5s"
    sleep 5
  done
  if [ "$ok" != "1" ]; then
    echo "    FAILED after $ATTEMPTS attempts"
    fail=1
    continue
  fi
  remote_size=$(ssh $SSH_OPTS "$HOST" "stat -c %s '$DEST/$f'" 2>/dev/null || echo -1)
  if [ "$remote_size" = "$local_size" ]; then
    echo "    size OK: $remote_size"
  else
    echo "    SIZE MISMATCH: local=$local_size remote=$remote_size"
    fail=1
  fi
done

echo
echo "=== size check summary ==="
ssh $SSH_OPTS "$HOST" "cd '$DEST' && ls -la" 2>/dev/null
exit "$fail"
