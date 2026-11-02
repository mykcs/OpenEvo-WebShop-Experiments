#!/usr/bin/env bash
set -euo pipefail
OUT="$1"
IMAGE="${CEILING1_IMAGE:-experiment-host/openevo-scientific:h144-runtime-v1}"  # legacy-revision-compat
install -d -o 1001 -g 1001 -m 750 "$OUT"
chown -R 1001:1001 "$OUT"
docker run --rm --network none --user 1001:1001 -v "$OUT:/workspace/out" --entrypoint sh "$IMAGE" -lc 'f=/workspace/out/.uid1001-write-smoke; : > "$f"; rm -f "$f"'
echo "SCIENCE_OUTPUT_WRITE_GATE=PASS path=$OUT uid=1001"
