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
#   ./yt-transcribe.sh package {firefox|chromium|all}
#                                         # sign/pack the browser extension for
#                                         # manual install — see below
#
# Override auto-detection if needed:
#   YT_TRANSCRIBE_GPU=cuda  ./yt-transcribe.sh start
#   YT_TRANSCRIBE_GPU=intel ./yt-transcribe.sh start
#
# Extension packaging:
#   - Source lives in ./extensions/yt-transcribe/ (this repo, git-tracked).
#   - Signed/packed output and the Chromium signing key live OUTSIDE the repo,
#     under ~/yt-transcribe/extensions/ (override with YT_TRANSCRIBE_HOME).
#   - Firefox signing needs AMO_JWT_ISSUER / AMO_JWT_SECRET — put them in a
#     gitignored .env.secrets file next to this script; the Firefox leg is
#     skipped (not failed) if they're unset.
#   - Chromium packing needs a chrome/chromium binary on PATH; no credentials
#     required. The first run generates and keeps a signing key permanently —
#     back it up (e.g. to pinet01); losing it breaks future updates.
#   - No auto-update server is set up (deliberately) — install each signed
#     build manually via about:addons / chrome://extensions.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_COMPOSE="${SCRIPT_DIR}/docker-compose.yml"

# ─── Extension packaging paths ─────────────────────────────────────────────────

EXT_SRC_DIR="${SCRIPT_DIR}/extensions/yt-transcribe"                 # git-tracked source
YT_TRANSCRIBE_HOME="${YT_TRANSCRIBE_HOME:-${HOME}/yt-transcribe}"    # deploy dir, not git-tracked
EXT_DEPLOY_DIR="${YT_TRANSCRIBE_HOME}/extensions"
EXT_DIST_DIR="${EXT_DEPLOY_DIR}/dist"
EXT_KEYS_DIR="${EXT_DEPLOY_DIR}/keys"
CHROMIUM_KEY="${EXT_KEYS_DIR}/yt-transcribe.pem"

# Optional local secrets (AMO API credentials) — gitignored, sourced if present
[ -f "${SCRIPT_DIR}/.env.secrets" ] && source "${SCRIPT_DIR}/.env.secrets"

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

# ─── Extension packaging helpers ───────────────────────────────────────────────

manifest_version() {
    local manifest="$1"
    if command -v jq &>/dev/null; then
        jq -r '.version' "${manifest}"
    else
        grep -m1 '"version"' "${manifest}" \
            | sed -E 's/.*"version"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/'
    fi
}

sign_extension() {
    [ -f "${EXT_SRC_DIR}/manifest.json" ] || die "Extension source not found at ${EXT_SRC_DIR}"

    if [ -z "${AMO_JWT_ISSUER:-}" ] || [ -z "${AMO_JWT_SECRET:-}" ]; then
        log "AMO_JWT_ISSUER / AMO_JWT_SECRET not set (see .env.secrets) — skipping Firefox signing."
        return 0
    fi
    command -v web-ext &>/dev/null || die "web-ext not found on PATH — install with: npm install --global web-ext"

    local version
    version="$(manifest_version "${EXT_SRC_DIR}/manifest.json")"
    log "Signing Firefox extension v${version}..."

    mkdir -p "${EXT_DIST_DIR}"
    web-ext sign \
        --source-dir="${EXT_SRC_DIR}" \
        --channel=unlisted \
        --api-key="${AMO_JWT_ISSUER}" \
        --api-secret="${AMO_JWT_SECRET}" \
        --artifacts-dir="${EXT_DIST_DIR}"

    log "Signed .xpi written to ${EXT_DIST_DIR}/"
}

pack_chromium_extension() {
    [ -f "${EXT_SRC_DIR}/manifest.json" ] || die "Extension source not found at ${EXT_SRC_DIR}"

    local chrome_bin
    chrome_bin="$(command -v google-chrome || command -v chromium-browser || command -v chromium || true)"
    [ -n "${chrome_bin}" ] || die "No Chrome/Chromium binary found on PATH."

    local version
    version="$(manifest_version "${EXT_SRC_DIR}/manifest.json")"

    mkdir -p "${EXT_DIST_DIR}" "${EXT_KEYS_DIR}"
    chmod 700 "${EXT_KEYS_DIR}"

    if [ ! -f "${CHROMIUM_KEY}" ]; then
        log "No existing signing key at ${CHROMIUM_KEY} — generating a NEW one."
        "${chrome_bin}" --pack-extension="${EXT_SRC_DIR}" --no-sandbox
        [ -f "${EXT_SRC_DIR}.pem" ] || die "Packing failed — no .pem produced."
        mv "${EXT_SRC_DIR}.pem" "${CHROMIUM_KEY}"
        chmod 600 "${CHROMIUM_KEY}"
        log "!! New key saved to ${CHROMIUM_KEY} — back this up NOW (e.g. to pinet01)."
        log "!! Losing it means you can never publish a trusted update under this ID again."
    else
        log "Packing Chromium extension v${version}..."
        "${chrome_bin}" --pack-extension="${EXT_SRC_DIR}" --pack-extension-key="${CHROMIUM_KEY}" --no-sandbox
    fi

    [ -f "${EXT_SRC_DIR}.crx" ] || die "Packing failed — no .crx produced."
    mv "${EXT_SRC_DIR}.crx" "${EXT_DIST_DIR}/yt-transcribe-${version}.crx"
    log "Packed: ${EXT_DIST_DIR}/yt-transcribe-${version}.crx"
    log "Install manually: drag onto chrome://extensions (Developer mode on)."
}

# ─── Git tagging ────────────────────────────────────────────────────────────

require_clean_worktree() {
    git -C "${SCRIPT_DIR}" diff --quiet --ignore-submodules -- . \
        || die "Uncommitted changes in the working tree — commit or stash before tagging"
    git -C "${SCRIPT_DIR}" diff --cached --quiet --ignore-submodules -- . \
        || die "Staged but uncommitted changes — commit before tagging"
}

tag_release() {
    local push=false
    [ "${1:-}" = "--push" ] && push=true

    [ -f "${EXT_SRC_DIR}/manifest.json" ] || die "Extension source not found at ${EXT_SRC_DIR}"
    require_clean_worktree

    local version tag
    version="$(manifest_version "${EXT_SRC_DIR}/manifest.json")"
    tag="v${version}"

    git -C "${SCRIPT_DIR}" rev-parse -q --verify "refs/tags/${tag}" &>/dev/null \
        && die "Tag ${tag} already exists — bump the version in manifest.json first"

    log "Tagging ${tag} at $(git -C "${SCRIPT_DIR}" rev-parse --short HEAD)"
    git -C "${SCRIPT_DIR}" tag -a "${tag}" -m "yt-transcribe ${tag}"

    if [ "${push}" = true ]; then
        log "Pushing tag ${tag} to origin..."
        git -C "${SCRIPT_DIR}" push origin "${tag}"
    else
        log "Tag created locally — push with: git push origin ${tag}"
    fi
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

cmd_package() {
    case "${1:-}" in
        firefox)  sign_extension ;;
        chromium) pack_chromium_extension ;;
        all)      sign_extension; pack_chromium_extension ;;
        *) die "Usage: $(basename "$0") package {firefox|chromium|all}" ;;
    esac
}

cmd_tag() {
    tag_release "$@"
}

cmd_release() {
    cmd_package all
    tag_release "$@"
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
    package) shift; cmd_package "$@" ;;
    tag)     shift; cmd_tag "$@" ;;
    release) shift; cmd_release "$@" ;;
    *)
        cat >&2 <<EOF
Usage: $(basename "$0") {start|stop|restart|build [args...]|clean|realclean|package {firefox|chromium|all}}

  start              Detect GPU backend and start the stack (up -d)
  stop               Stop the stack (down)
  restart            stop, then start
  build [args...]    Build images; extra args passed through (e.g. --no-cache)
  clean              docker system prune -f (host-wide — see warning)
  realclean          stop, then clean
  package TARGET     Sign/pack the browser extension for manual install:
                       firefox  — AMO-signed .xpi (needs AMO_JWT_ISSUER/SECRET)
                       chromium — self-signed .crx (needs chrome/chromium on PATH)
                       all      — both
  tag [--push]       Tag HEAD as v<manifest version> (fails on dirty tree or duplicate tag)
  release [--push]   package all, then tag (fails closed if either step fails)

GPU backend is auto-detected (nvidia-smi -> cuda, /dev/dri/renderD128 -> intel).
Override with: YT_TRANSCRIBE_GPU=cuda|intel
Extension deploy dir defaults to ~/yt-transcribe/extensions — override with YT_TRANSCRIBE_HOME.
EOF
        exit 1
        ;;
esac
