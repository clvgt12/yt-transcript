#!/usr/bin/env bash
# yt_transcribe.sh — Download YouTube audio, transcribe with Whisper, optionally summarize with Ollama
#
# Usage:
#   ./yt_transcribe.sh <YouTube_URL> [options]
#
# Arguments:
#   YouTube_URL              Full YouTube video URL (required)
#
# Options:
#   --whisper=MODEL          Whisper model: tiny, base, small, medium, large
#                            (default: base)
#   --summarize[=MODEL]      Enable Ollama summarization. Optionally specify
#                            the Ollama model name (default: gemma3:1b)
#   -h, --help               Show this help message and exit
#
# Dependencies:
#   - yt-dlp        (snap: yt-dlp)
#   - ffmpeg        (apt:  ffmpeg)
#   - whisper       (pip:  openai-whisper, inside venv at ~/venvs/openai-whisper)
#   - docker        (apt:  docker.io or docker-ce)  [required only with --summarize]
#   - jq            (apt:  jq)                      [required only with --summarize]
#
# Output (written to ~/Downloads/<video_id>/):
#   <title>.mp3     Downloaded audio
#   <title>.txt     Whisper transcript
#   <title>.md      Ollama summary in markdown (only with --summarize)
#
# Notes:
#   - Whisper model weights are cached in ~/.cache/whisper/ on first use
#   - GPU (CUDA) acceleration is used automatically if available
#   - tiny/base/small models run on GPU; medium/large run on CPU (GTX 1050 Ti
#     has insufficient VRAM for larger models alongside the KDE desktop stack)
#   - PyTorch 2.2.0+cu118 with numpy<2 required for GTX 1050 Ti (Pascal/sm_61)
#   - Download is skipped if an MP3 already exists in the output directory
#   - Ollama runs in a temporary container on port 11435, removed after use
#   - Ollama model weights are reused from the existing 'ollama' Docker volume
#
# Examples:
#   ./yt_transcribe.sh "https://youtube.com/watch?v=XXXXX"
#   ./yt_transcribe.sh "https://youtube.com/watch?v=XXXXX" --whisper=medium
#   ./yt_transcribe.sh "https://youtube.com/watch?v=XXXXX" --summarize
#   ./yt_transcribe.sh "https://youtube.com/watch?v=XXXXX" --summarize=qwen3:1.7b
#   ./yt_transcribe.sh "https://youtube.com/watch?v=XXXXX" --whisper=small --summarize=gemma3:1b
#
# Change history:
#   See git log for revision history

set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────

VENV_PATH="${HOME}/venvs/openai-whisper"
OUTPUT_BASE="${HOME}/Downloads"
DEFAULT_WHISPER_MODEL="base"
DEFAULT_OLLAMA_MODEL="gemma3:1b"
YT_DLP_BIN="/snap/bin/yt-dlp"

# Models that fit in VRAM alongside the KDE desktop stack (~1.5 GB overhead)
GPU_MODELS="tiny base small"

# Ollama container settings
OLLAMA_IMAGE="ollama/ollama"
OLLAMA_CONTAINER="yt-transcribe-ollama-$$"   # $$ = PID, ensures uniqueness
OLLAMA_HOST_PORT="11435"                       # Dedicated port, avoids conflict with existing Ollama
OLLAMA_VOLUME="ollama"                         # Reuse existing ollama Docker volume
OLLAMA_URL="http://localhost:${OLLAMA_HOST_PORT}"

# ─── Argument handling ────────────────────────────────────────────────────────

usage() {
    grep '^#' "$0" | grep -v '#!/' | sed 's/^# \{0,1\}//'
    exit 0
}

YT_URL=""
WHISPER_MODEL="$DEFAULT_WHISPER_MODEL"
OLLAMA_MODEL="$DEFAULT_OLLAMA_MODEL"
SUMMARIZE=false

for arg in "$@"; do
    case "$arg" in
        --whisper=*)
            WHISPER_MODEL="${arg#--whisper=}"
            ;;
        --summarize=*)
            SUMMARIZE=true
            OLLAMA_MODEL="${arg#--summarize=}"
            ;;
        --summarize)
            SUMMARIZE=true
            ;;
        --help|-h)
            usage
            ;;
        --*)
            echo "Error: Unknown option '${arg}'" >&2
            echo "       Run with --help for usage." >&2
            exit 1
            ;;
        *)
            if [[ -z "$YT_URL" ]]; then
                YT_URL="$arg"
            else
                echo "Error: Unexpected argument '${arg}'" >&2
                echo "       Run with --help for usage." >&2
                exit 1
            fi
            ;;
    esac
done

if [[ -z "$YT_URL" ]]; then
    echo "Error: YouTube URL is required." >&2
    echo "       Run with --help for usage." >&2
    exit 1
fi

# Validate Whisper model name
VALID_MODELS="tiny base small medium large"
if ! echo "$VALID_MODELS" | grep -qw "$WHISPER_MODEL"; then
    echo "Error: Invalid Whisper model '${WHISPER_MODEL}'. Choose from: ${VALID_MODELS}" >&2
    exit 1
fi

# Select Whisper compute device based on model size
if echo "$GPU_MODELS" | grep -qw "$WHISPER_MODEL"; then
    WHISPER_DEVICE="cuda"
else
    WHISPER_DEVICE="cpu"
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

if [[ "$SUMMARIZE" == "true" ]]; then
    check_dep "docker"   "Install with: sudo apt install docker.io"
    check_dep "curl"     "Install with: sudo apt install curl"
    check_dep "jq"       "Install with: sudo apt install jq"
fi

if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
    echo "Error: Whisper venv not found at ${VENV_PATH}" >&2
    echo "       Create it with: python3 -m venv ${VENV_PATH}" >&2
    echo "       Then: source ${VENV_PATH}/bin/activate && pip install -r ${VENV_PATH}/requirements.txt" >&2
    exit 1
fi

# ─── Cleanup trap — always remove Ollama container on exit ───────────────────

OLLAMA_STARTED=false

cleanup() {
    if [[ "$OLLAMA_STARTED" == "true" ]]; then
        echo ""
        echo "==> Stopping and removing Ollama container (${OLLAMA_CONTAINER})..."
        docker stop "$OLLAMA_CONTAINER" &>/dev/null || true
        docker rm   "$OLLAMA_CONTAINER" &>/dev/null || true
        echo "==> Ollama container removed."
    fi
}
trap cleanup EXIT

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

echo "==> Video ID      : ${VIDEO_ID}"
echo "==> Title         : ${VIDEO_TITLE}"
echo "==> Output dir    : ${OUTPUT_DIR}"
echo "==> Whisper model : ${WHISPER_MODEL} (${WHISPER_DEVICE})"
if [[ "$SUMMARIZE" == "true" ]]; then
    echo "==> Summarize     : yes (${OLLAMA_MODEL})"
else
    echo "==> Summarize     : no"
fi
if [[ "$WHISPER_DEVICE" == "cpu" ]]; then
    echo "    Note: medium/large models exceed available VRAM — Whisper falling back to CPU"
fi

# ─── Download audio (skip if MP3 already exists) ─────────────────────────────

echo ""
EXISTING_MP3=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*.mp3" | head -1)

if [[ -n "$EXISTING_MP3" ]]; then
    echo "==> Audio already exists, skipping download."
    echo "==> Audio found : ${EXISTING_MP3}"
    AUDIO_FILE="$EXISTING_MP3"
else
    echo "==> Downloading audio..."

    "$YT_DLP_BIN" \
        --extract-audio \
        --audio-format mp3 \
        --audio-quality 0 \
        --output "${OUTPUT_DIR}/%(title)s.%(ext)s" \
        "$YT_URL"

    AUDIO_FILE=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*.mp3" | head -1)

    if [[ -z "$AUDIO_FILE" ]]; then
        echo "Error: Audio download failed — no MP3 found in ${OUTPUT_DIR}" >&2
        exit 1
    fi

    echo "==> Audio saved : ${AUDIO_FILE}"
fi

# ─── Transcribe with Whisper ──────────────────────────────────────────────────

echo ""
echo "==> Activating Whisper venv..."
# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"

echo "==> Transcribing with model '${WHISPER_MODEL}' on ${WHISPER_DEVICE}..."
whisper "$AUDIO_FILE" \
    --model "$WHISPER_MODEL" \
    --device "$WHISPER_DEVICE" \
    --output_dir "$OUTPUT_DIR" \
    --output_format txt \
    --verbose False

deactivate

TRANSCRIPT_FILE=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*.txt" | head -1)

if [[ -z "$TRANSCRIPT_FILE" ]]; then
    echo "Error: Transcription failed — no .txt file found in ${OUTPUT_DIR}" >&2
    exit 1
fi

echo "==> Transcript  : ${TRANSCRIPT_FILE}"

# ─── Summarize with Ollama (optional) ────────────────────────────────────────

SUMMARY_FILE=""

if [[ "$SUMMARIZE" == "true" ]]; then

    echo ""
    echo "==> Starting Ollama container (${OLLAMA_CONTAINER}) on port ${OLLAMA_HOST_PORT}..."

    docker run -d \
        --name "$OLLAMA_CONTAINER" \
        --gpus all \
        -p "${OLLAMA_HOST_PORT}:11434" \
        -v "${OLLAMA_VOLUME}:/root/.ollama" \
        "$OLLAMA_IMAGE" &>/dev/null

    OLLAMA_STARTED=true

    # Wait for Ollama API to become ready (up to 30 seconds)
    echo "==> Waiting for Ollama API to be ready..."
    READY=false
    for i in $(seq 1 30); do
        if curl -sf "${OLLAMA_URL}/api/tags" &>/dev/null; then
            READY=true
            break
        fi
        sleep 1
    done

    if [[ "$READY" != "true" ]]; then
        echo "Error: Ollama API did not become ready within 30 seconds." >&2
        exit 1
    fi

    echo "==> Ollama ready."

    # Pull model if not already cached in the volume
    echo "==> Checking for model '${OLLAMA_MODEL}'..."
    MODEL_EXISTS=$(curl -sf "${OLLAMA_URL}/api/tags" | grep -c "\"${OLLAMA_MODEL}\"" || true)

    if [[ "$MODEL_EXISTS" -eq 0 ]]; then
        echo "==> Pulling model '${OLLAMA_MODEL}' (first use — cached to Docker volume)..."
        curl -sf -X POST "${OLLAMA_URL}/api/pull" \
            -H "Content-Type: application/json" \
            -d "{\"name\": \"${OLLAMA_MODEL}\"}" | grep -v '^$' | tail -1
        echo ""
    else
        echo "==> Model '${OLLAMA_MODEL}' already cached."
    fi

    # Build and send summarization prompt
    echo "==> Summarizing transcript with '${OLLAMA_MODEL}'..."

    TRANSCRIPT_TEXT=$(cat "$TRANSCRIPT_FILE")
    SUMMARY_FILE="${OUTPUT_DIR}/${SAFE_TITLE}_summary.md"

    PROMPT="You are a professional analyst. Read the following transcript carefully and produce a structured summary in Markdown format with exactly three sections:

## Summary
Write a concise summary of 3-5 sentences covering the core subject and conclusions.

## Key Points
Bullet list of the most important facts, arguments, or events from the transcript.

## Takeaways
Bullet list of the key insights, implications, or action items a reader should walk away with.

Use clean Markdown formatting. Be precise and objective. Do not editorialize.

---
TRANSCRIPT:
${TRANSCRIPT_TEXT}"

    RESPONSE=$(curl -sf -X POST "${OLLAMA_URL}/api/generate" \
        -H "Content-Type: application/json" \
        -d "$(jq -n --arg model "$OLLAMA_MODEL" --arg prompt "$PROMPT" \
            '{model: $model, prompt: $prompt, stream: false}')")

    # Write markdown summary file with header metadata
    {
        echo "# ${VIDEO_TITLE}"
        echo ""
        echo "_Source: ${YT_URL}_"
        echo ""
        echo "_Transcribed with Whisper \`${WHISPER_MODEL}\` — Summarized with Ollama \`${OLLAMA_MODEL}\`_"
        echo ""
        echo "---"
        echo ""
        echo "$RESPONSE" | jq -r '.response'
    } > "$SUMMARY_FILE"

    echo "==> Summary     : ${SUMMARY_FILE}"

fi

# ─── Report output ────────────────────────────────────────────────────────────

echo ""
echo "==> Done."
echo "    Audio      : ${AUDIO_FILE}"
echo "    Transcript : ${TRANSCRIPT_FILE}"
if [[ -n "$SUMMARY_FILE" ]]; then
    echo "    Summary    : ${SUMMARY_FILE}"
fi
echo "    All outputs: ${OUTPUT_DIR}"
