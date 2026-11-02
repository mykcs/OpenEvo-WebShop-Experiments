#!/usr/bin/env bash

# Fetch the three WebShop small-dataset JSON files exclusively through
# api.github.com. This avoids the lab Mac's unstable routes to Hugging Face and
# Google Drive and avoids GitHub Actions/billing entirely.

set -euo pipefail

TARGET_DIR="${1:?usage: fetch_webshop_data_via_github_api.sh TARGET_DIR}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREPARER="$SCRIPT_DIR/prepare_webshop_data.py"
SOURCE_REPO="WooooDyy/AgentGym"
SOURCE_REV="3ef9235d23e68e7c2920c5422ad957dc8ced5c6c"
SOURCE_PREFIX="agentenv-webshop/webshop/data"
SOURCE_LABEL="github-api:${SOURCE_REPO}@${SOURCE_REV}"
TMP_DIR="$TARGET_DIR/.github-api-webshop-data.$$"

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "WEBSHOP_DATA_GITHUB_API_MISSING_COMMAND=$1" >&2
    exit 2
  }
}
for cmd in gh python3 mkdir rm mv; do need_cmd "$cmd"; done

mkdir -p "$TARGET_DIR"
rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"
trap 'rm -rf "$TMP_DIR"' EXIT

fetch_one() {
  local name="$1"
  local path="$SOURCE_PREFIX/$name"
  local metadata_endpoint="repos/$SOURCE_REPO/contents/$path?ref=$SOURCE_REV"
  local blob_sha size

  echo "WEBSHOP_DATA_GITHUB_API_LOOKUP name=$name source=$SOURCE_REPO rev=$SOURCE_REV"
  blob_sha="$(gh api "$metadata_endpoint" --jq .sha 2>/dev/null)" || {
    echo "WEBSHOP_DATA_GITHUB_API_METADATA_FAILED name=$name" >&2
    return 1
  }
  size="$(gh api "$metadata_endpoint" --jq .size 2>/dev/null)" || size="unknown"
  [[ -n "$blob_sha" && "$blob_sha" != "null" ]] || {
    echo "WEBSHOP_DATA_GITHUB_API_BLOB_SHA_MISSING name=$name" >&2
    return 1
  }
  echo "WEBSHOP_DATA_GITHUB_API_BLOB name=$name sha=$blob_sha bytes=$size"

  if ! gh api "repos/$SOURCE_REPO/git/blobs/$blob_sha" --jq .content \
      | python3 -c 'import base64,sys; sys.stdout.buffer.write(base64.b64decode(sys.stdin.read()))' \
      > "$TMP_DIR/$name"; then
    echo "WEBSHOP_DATA_GITHUB_API_BLOB_DOWNLOAD_FAILED name=$name sha=$blob_sha" >&2
    return 1
  fi
}

for name in items_shuffle_1000.json items_ins_v2_1000.json items_human_ins.json; do
  fetch_one "$name" || exit 3
done

if ! python3 "$PREPARER" --validate-only --write-manifest --source "$SOURCE_LABEL" "$TMP_DIR"; then
  echo "WEBSHOP_DATA_GITHUB_API_VALIDATE_FAILED" >&2
  exit 4
fi

for name in items_shuffle_1000.json items_ins_v2_1000.json items_human_ins.json MANIFEST.json; do
  mv -f "$TMP_DIR/$name" "$TARGET_DIR/$name"
done

if ! python3 "$PREPARER" --validate-only "$TARGET_DIR"; then
  echo "WEBSHOP_DATA_GITHUB_API_CACHE_VALIDATE_FAILED" >&2
  exit 5
fi

echo "WEBSHOP_DATA_GITHUB_API_READY=$TARGET_DIR"
