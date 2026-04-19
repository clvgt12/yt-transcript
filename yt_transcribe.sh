#!/usr/bin/env bash
# yt_transcribe.sh — Download YouTube audio and transcribe with Whisper
#
# Usage:
#   ./yt_transcribe.sh <YouTube_URL> [whisper_model]
#
# Arguments:
#   YouTube_URL     Full YouTube video URL (required)
#   whisper_model   Whisper model to use: tiny, base, small, medium, large
#                   Defaults to: base
#
# Dependencies:
#   - yt-dlp        (snap: yt-dlp)
#   - ffmpeg        (apt:  ffmpeg)
#   - whisper       (pip:  openai-whisper, inside venv at ~/venvs/openai-whisper)
#
# Output:
#   Audio and transcript files are written to ~/yt_transcribe/<video_id>/
#
# Notes:
#   - Whisper model weights are cached in ~/.cache/whisper/ on first use
#   - GPU (CUDA) acceleration is used automatically if available
#   - PyTorch 2.2.0+cu118 with numpy<2 required for GTX 1050 Ti (Pascal/sm_61)
#
# Change history:
#   See git log for revision history

set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────

VENV_PATH="${HOME}/venvs/openai-whisper"
OUTPUT_BASE="${HOME}/yt_transcribe"
DEFAULT_MODEL="base"
YT_DLP_BIN="/snap/bin/yt-dlp"

# ─── Argument handling ────────────────────────────────────────────────────────

usage() {
    grep '^#' "$0" | grep -v '#!/' | sed 's/^# \{0,1\}//'
    exit 1
}

if [[ $# -lt 1 ]]; then
    echo "Error: YouTube URL is required." >&2
    usage
fi

YT_URL="$1"
WHISPER_MODEL="${2:-$DEFAULT_MODEL}"

# Validate model name
VALID_MODELS="tiny base small medium large"
if ! echo "$VALID_MODELS" | grep -qw "$WHISPER_MODEL"; then
    echo "Error: Invalid model '${WHISPER_MODEL}'. Choose from: ${VALID_MODELS}" >&2
    exit 1
fi

# ─── Dependency checks ────────────────────────────────────────────────────────

check_dep() {
    local bin="$1"
    local hint="$2"
    if ! command -v "$bin" &>/dev/null && [[ ! -x "$bin" ]]; then
        echo "Error: '${bin}' not found. ${hint}" >&2
        exit 1
    fi
}

check_dep "$YT_DLP_BIN"  "Install with: sudo snap install yt-dlp"
check_dep "ffmpeg"        "Install with: sudo apt install ffmpeg"

if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
    echo "Error: Whisper venv not found at ${VENV_PATH}" >&2
    echo "       Create it with: python3 -m venv ${VENV_PATH}" >&2
    echo "       Then: source ${VENV_PATH}/bin/activate && pip install -r ${VENV_PATH}/requirements.txt" >&2
    exit 1
fi

# ─── Resolve video ID for output directory naming ─────────────────────────────

echo "==> Resolving video metadata..."
VIDEO_ID=$("$YT_DLP_BIN" --print id "$YT_URL" 2>/dev/null) || {
    echo "Error: Could not resolve video ID. Check the URL or yt-dlp version." >&2
    exit 1
}

VIDEO_TITLE=$("$YT_DLP_BIN" --print title "$YT_URL" 2>/dev/null || echo "unknown_title")

# Sanitize title for filesystem use
SAFE_TITLE=$(echo "$VIDEO_TITLE" | tr -cd '[:alnum:] _-' | tr ' ' '_' | cut -c1-60)

OUTPUT_DIR="${OUTPUT_BASE}/${VIDEO_ID}"
mkdir -p "$OUTPUT_DIR"

AUDIO_FILE="${OUTPUT_DIR}/${SAFE_TITLE}.mp3"

echo "==> Video ID    : ${VIDEO_ID}"
echo "==> Title       : ${VIDEO_TITLE}"
echo "==> Output dir  : ${OUTPUT_DIR}"
echo "==> Whisper model: ${WHISPER_MODEL}"

# ─── Download audio ───────────────────────────────────────────────────────────

echo ""
echo "==> Downloading audio..."

"$YT_DLP_BIN" \
    --extract-audio \
    --audio-format mp3 \
    --audio-quality 0 \
    --output "${OUTPUT_DIR}/%(title)s.%(ext)s" \
    "$YT_URL"

# Locate the downloaded MP3 (title may differ slightly from our sanitized name)
AUDIO_FILE=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*.mp3" | head -1)

if [[ -z "$AUDIO_FILE" ]]; then
    echo "Error: Audio download failed — no MP3 found in ${OUTPUT_DIR}" >&2
    exit 1
fi

echo "==> Audio saved : ${AUDIO_FILE}"

# ─── Transcribe with Whisper ──────────────────────────────────────────────────

echo ""
echo "==> Activating Whisper venv..."
# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"

echo "==> Transcribing with model '${WHISPER_MODEL}'..."
whisper "$AUDIO_FILE" \
    --model "$WHISPER_MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --output_format txt \
    --verbose False

deactivate

# ─── Report output ────────────────────────────────────────────────────────────

TRANSCRIPT_FILE=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*.txt" | head -1)

echo ""
echo "==> Done."
echo "    Audio      : ${AUDIO_FILE}"
echo "    Transcript : ${TRANSCRIPT_FILE:-'(not found — check for errors above)'}"
echo "    All outputs: ${OUTPUT_DIR}"
