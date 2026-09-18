#!/usr/bin/env bash
#
# yt-transcribe.sh — start/stop/build/clean the yt-transcribe-web Docker Compose stack.
#
# Automatically detects the host's GPU backend (NVIDIA/CUDA vs Intel iGPU/OpenVINO)
# and selects the matching compose override file — docker-compose.cuda.yml on
# kamakazi, docker-compose.intel.yml on tepache — so the same script works
# unmodified on either host.
#
# Usage:
#   ./yt-transcribe.sh start              # up -d, detected GPU backend
#   ./yt-transcribe.sh stop               # down
#   ./yt-transcribe.sh restart            # stop, then start
#   ./yt-transcribe.sh build [args...]    # build; extra args passed through
#                                         #   e.g. ./yt-transcribe.sh build --no-cache
#   ./yt-transcribe.sh clean              # docker system prune -f (see warning below)
#
# Override auto-detection if needed:
#   YT_TRANSCRIBE_GPU=cuda  ./yt-transcribe.sh start
#   YT_TRANSCRIBE_GPU=intel ./yt-transcribe.sh start

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_COMPOSE="${SCRIPT_DIR}/docker-compose.yml"

# ─── Logging helpers ──────────────────────────────────────────────────────────

log()  { printf '[%s] %s\n' "$(date '+%Y-%m-%dT%H:%M:%S')" "$*"; }
die()  { printf '[%s] ERROR: %s\n' "$(date '+%Y-%m-%dT%H:%M:%S')" "$*" >&2; exit 1; }

# ─── GPU backend detection ─────────────────────────────────────────────────────

detect_gpu_backend() {
    # Explicit override wins — useful for a host detection gets wrong, or a
    # future third backend.
    if [ -n "${YT_TRANSCRIBE_GPU:-}" ]; then
        echo "${YT_TRANSCRIBE_GPU}"
        return
    fi

    # nvidia-smi succeeding means the NVIDIA driver stack is actually loaded
    # and usable — not just that an NVIDIA card is physically present.
    if command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null; then
        echo "cuda"
        return
    fi

    # /dev/dri/renderD128 is the Intel (or any) GPU render node; on our two
    # target hosts its presence without a working nvidia-smi means Intel iGPU.
    if [ -e /dev/dri/renderD128 ]; then
        echo "intel"
        return
    fi

    echo "none"
}

compose_files_for_backend() {
    case "$1" in
        cuda)  echo "${BASE_COMPOSE} ${SCRIPT_DIR}/docker-compose.cuda.yml" ;;
        intel) echo "${BASE_COMPOSE} ${SCRIPT_DIR}/docker-compose.intel.yml" ;;
        *)     die "Unsupported GPU backend '$1' — expected 'cuda' or 'intel' (set YT_TRANSCRIBE_GPU to override detection)" ;;
    esac
}

# ─── Compose file args, resolved once at startup for start/stop/restart/build ──

GPU_BACKEND="$(detect_gpu_backend)"

compose_args() {
    local files
    files="$(compose_files_for_backend "${GPU_BACKEND}")"
    local args=()
    for f in ${files}; do
        args+=("-f" "${f}")
    done
    printf '%s\n' "${args[@]}"
}

run_compose() {
    local args
    mapfile -t args < <(compose_args)
    docker compose "${args[@]}" "$@"
}

# ─── Subcommands ──────────────────────────────────────────────────────────────

cmd_start() {
    log "GPU backend: ${GPU_BACKEND} (override with YT_TRANSCRIBE_GPU=cuda|intel)"
    [ "${GPU_BACKEND}" = "none" ] && die "No supported GPU backend detected — nvidia-smi failed and /dev/dri/renderD128 not found"
    log "Starting stack..."
    run_compose up -d
}

cmd_stop() {
    log "GPU backend: ${GPU_BACKEND}"
    log "Stopping stack..."
    run_compose down
}

cmd_restart() {
    cmd_stop
    cmd_start
}

cmd_build() {
    log "GPU backend: ${GPU_BACKEND} (override with YT_TRANSCRIBE_GPU=cuda|intel)"
    [ "${GPU_BACKEND}" = "none" ] && die "No supported GPU backend detected — nvidia-smi failed and /dev/dri/renderD128 not found"
    log "Building images...${*:+ (args: $*)}"
    run_compose build "$@"
}

cmd_clean() {
    log "WARNING: 'docker system prune -f' removes ALL stopped containers,"
    log "         dangling images, and unused networks/build cache on this"
    log "         host — not just yt-transcribe-web's. On a host running other"
    log "         Docker workloads (minikube, other Compose projects), this"
    log "         can remove more than you expect."
    log "Pruning Docker system..."
    docker system prune -f
}

cmd_realclean() {
    cmd_stop
    cmd_clean
}

# ─── Entry point ──────────────────────────────────────────────────────────────

[ -f "${BASE_COMPOSE}" ] || die "docker-compose.yml not found in ${SCRIPT_DIR} — run this script from the repo root"

case "${1:-}" in
    start)   cmd_start ;;
    stop)    cmd_stop ;;
    restart) cmd_restart ;;
    build)   shift; cmd_build "$@" ;;
    clean)   cmd_clean ;;
    realclean) cmd_realclean ;;
    *)
        cat >&2 <<EOF
Usage: $(basename "$0") {start|stop|restart|build [args...]|clean|realclean}

  start              Detect GPU backend and start the stack (up -d)
  stop               Stop the stack (down)
  restart            stop, then start
  build [args...]    Build images; extra args passed through (e.g. --no-cache)
  clean              docker system prune -f (host-wide — see warning)
  realclean          stop, then clean

GPU backend is auto-detected (nvidia-smi -> cuda, /dev/dri/renderD128 -> intel).
Override with: YT_TRANSCRIBE_GPU=cuda|intel
EOF
        exit 1
        ;;
esac
