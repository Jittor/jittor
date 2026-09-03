#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Legacy converter service runner.

Usage:
  CONVERTER_TLS_DIR=/etc/letsencrypt converter_server.sh --run

This script installs the latest published Jittor package in a long-running
container. Review it and its network/TLS settings before every deployment.
EOF
}

if [[ "${1:-}" != "--run" ]]; then
    usage >&2
    exit 2
fi

for tool in docker timeout; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'required service tool is unavailable: %s\n' "$tool" >&2
        exit 1
    fi
done

TLS_DIR="${CONVERTER_TLS_DIR:-}"
if [[ -z "$TLS_DIR" || ! -d "$TLS_DIR" ]]; then
    printf 'CONVERTER_TLS_DIR must name a readable TLS certificate directory.\n' >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LAB_ROOT="${JITTOR_LAB_ROOT:-$(cd "$REPO_ROOT/.." && pwd)/jittor-lab}"
STATE_DIR="$LAB_ROOT/_state/tools/converter-server"
IMAGE="${CONVERTER_IMAGE:-jittor/converter_server:legacy}"
HOST_PORT="${CONVERTER_HOST_PORT:-58187}"
CONTAINER_TTL="${CONVERTER_TTL:-24h}"
RESTART_DELAY="${CONVERTER_RESTART_DELAY:-10}"

mkdir -p "$STATE_DIR"
DOCKERFILE="$STATE_DIR/Dockerfile"
cat >"$DOCKERFILE" <<'EOF'
FROM jittor/jittor

RUN python -m pip install --no-cache-dir flask
EOF

docker build --tag "$IMAGE" --file "$DOCKERFILE" "$REPO_ROOT"

while true; do
    set +e
    timeout --foreground "$CONTAINER_TTL" \
        docker run --rm --init \
        --memory 16g \
        --cpus 8 \
        --publish "0.0.0.0:${HOST_PORT}:5000" \
        --volume "$TLS_DIR:/https:ro" \
        "$IMAGE" \
        bash -lc 'python -m pip install --no-cache-dir --upgrade jittor && python -m jittor.selftest && FLASK_APP=jittor.compat.converter_server python -m flask run --cert=/https/live/randonl.me/fullchain.pem --key=/https/live/randonl.me/privkey.pem --host=0.0.0.0'
    status=$?
    set -e
    if [[ "$status" -ne 0 && "$status" -ne 124 ]]; then
        printf 'converter container exited unexpectedly with status %s\n' "$status" >&2
        exit "$status"
    fi
    sleep "$RESTART_DELAY"
done
