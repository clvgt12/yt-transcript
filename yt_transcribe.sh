#!/usr/bin/env bash
# yt_transcribe.sh — Download YouTube audio, transcribe with Whisper, optionally summarize with Ollama
# Container-aware version: respects YT_DLP_BIN, OUTPUT_BASE, OLLAMA_URL env vars
#
# Usage:
#   ./yt_transcribe.sh <YouTube_URL> [options]
#
# Options:
#   --whisper=MODEL          Whisper model: tiny, base, small, medium, large (default: small)
#   --summarize[=MODEL]      Enable Ollama summarization, optionally specify model (default: qwen3:1.7b)
#   --force-whisper          Skip subtitle check, always use Whisper for transcription
#   --force-local-summary    Skip Ollama cloud API, summarize with local model only
#   -h, --help               Show this help and exit
#
# Transcription strategy (in order of preference):
#   1. Human-written YouTube subtitles  (fastest, highest quality when available)
#   2. Auto-generated YouTube subtitles (fast, variable quality)
#   3. Whisper local inference          (slowest, most reliable fallback)
#
# Environment variables (set by Docker Compose):
#   YT_DLP_BIN               Path to yt-dlp binary (default: /usr/local/bin/yt-dlp)
#   OUTPUT_BASE              Output root directory (default: /outputs)
#   VENV_PATH                Python venv path (default: ~/venvs/openai-whisper)
#   OLLAMA_URL               Ollama API base URL (default: http://ollama:11434)
#   OLLAMA_HOST_PORT         Ollama container port (default: 11434)
#
# Change history:
#   See git log for revision history

set -euo pipefail

SCRIPT_START=$(date +%s)

# ─── VTT to plain text converter ─────────────────────────────────────────────
# Defined as a function — strips WebVTT headers, timestamps, and duplicate lines

_vtt_to_txt() {
    local vtt_in="$1"
    local txt_out="$2"

    python3 - "$vtt_in" "$txt_out" << 'INNEREOF'
import re, sys

vtt_path = sys.argv[1]
txt_path = sys.argv[2]

with open(vtt_path, "r", encoding="utf-8") as f:
    raw = f.read()

# Remove WEBVTT header block
raw = re.sub(r"^WEBVTT[^\n]*\n.*?\n\n", "", raw, count=1, flags=re.DOTALL)

# Split into cue blocks on double newlines — blank lines are paragraph signals
blocks = re.split(r"\n{2,}", raw)

sentences     = []
para_breaks   = set()
prev_line     = None
prev_sent_end = False

for block in blocks:
    if not block.strip():
        continue
    lines = block.strip().splitlines()

    # Strip cue ID (pure digits)
    if lines and re.match(r"^\s*\d+\s*$", lines[0]):
        lines = lines[1:]
    # Strip timestamp line
    if lines and re.match(r"\d{2}:\d{2}[\d:,.]+\s*-->\s*\d{2}:\d{2}", lines[0]):
        lines = lines[1:]
    # Skip NOTE blocks
    if lines and lines[0].startswith("NOTE"):
        continue

    # Clean VTT/HTML tags and entities
    cleaned = []
    for line in lines:
        line = re.sub(r"<[^>]+>", "", line)
        line = re.sub(r"&amp;", "&", line)
        line = re.sub(r"&lt;",  "<", line)
        line = re.sub(r"&gt;",  ">", line)
        line = line.strip()
        if line:
            cleaned.append(line)

    if not cleaned:
        continue

    for line in cleaned:
        if line == prev_line:
            continue
        sentences.append(line)
        # Record paragraph break at this sentence if the PREVIOUS
        # sentence ended the cue with sentence-final punctuation
        if prev_sent_end:
            para_breaks.add(len(sentences) - 1)
        prev_line     = line
        prev_sent_end = bool(re.search(r"[.!?]\s*$", line))

# Merge continuation fragments (same cue, no sentence-final punct, lowercase next)
merged = []
for i, sent in enumerate(sentences):
    if (merged
            and not re.search(r"[.!?,;:]\s*$", merged[-1])
            and i not in para_breaks
            and sent[:1].islower()):
        merged[-1] = merged[-1].rstrip() + " " + sent
    else:
        merged.append(sent)

# Build paragraphs — break when VTT had a blank-line boundary AND
# current sentence ends with .!? AND next sentence starts with capital
paragraphs = []
current    = []

for i, sent in enumerate(merged):
    current.append(sent)
    is_sent_end    = bool(re.search(r"[.!?]\s*$", sent))
    next_is_cap    = (i + 1 < len(merged) and merged[i + 1][:1].isupper())
    has_vtt_break  = (i + 1) in para_breaks

    if is_sent_end and next_is_cap and has_vtt_break:
        paragraphs.append(" ".join(current))
        current = []

if current:
    paragraphs.append(" ".join(current))

output = "\n\n".join(paragraphs)
output = re.sub(r" +", " ", output).strip()

with open(txt_path, "w", encoding="utf-8") as f:
    f.write(output + "\n")

print(f"Converted {len(merged)} lines → {len(paragraphs)} paragraphs → {txt_path}")
INNEREOF
}

# Export function so it's available in the script scope
export -f _vtt_to_txt 2>/dev/null || true


# ─── Configuration ────────────────────────────────────────────────────────────

VENV_PATH="${VENV_PATH:-${HOME}/venvs/openai-whisper}"
OUTPUT_BASE="${OUTPUT_BASE:-${HOME}/Downloads/yt_transcribe}"
DEFAULT_WHISPER_MODEL="small"
DEFAULT_OLLAMA_MODEL="qwen3:1.7b"
YT_DLP_BIN="${YT_DLP_BIN:-/snap/bin/yt-dlp}"

# Ollama local container settings
OLLAMA_IMAGE="ollama/ollama"
OLLAMA_CONTAINER="yt-transcribe-ollama-$$"
OLLAMA_HOST_PORT="${OLLAMA_HOST_PORT:-11435}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:${OLLAMA_HOST_PORT}}"
OLLAMA_EXTERNAL="${OLLAMA_URL:-}"
OLLAMA_VOLUME="ollama"

# Ollama cloud settings
OLLAMA_CLOUD_URL="https://ollama.com/api"
DEFAULT_OLLAMA_CLOUD_MODEL="gpt-oss:20b"
OLLAMA_CLOUD_MODEL="${OLLAMA_CLOUD_MODEL:-${DEFAULT_OLLAMA_CLOUD_MODEL}}"
# OLLAMA_API_KEY read from environment — not set here

# Models that fit in VRAM alongside the KDE desktop stack (~1.5 GB overhead)
GPU_MODELS="tiny base small"

# ─── Argument handling ────────────────────────────────────────────────────────

usage() {
    sed -n '/^# yt_transcribe/,/^# Change history/{p}' "$0" \
        | sed 's/^# \{0,1\}//'
    exit 0
}

YT_URL=""
WHISPER_MODEL="$DEFAULT_WHISPER_MODEL"
OLLAMA_MODEL="$DEFAULT_OLLAMA_MODEL"
SUMMARIZE=false
FORCE_WHISPER=false
FORCE_LOCAL_SUMMARY=false

for arg in "$@"; do
    case "$arg" in
        --whisper=*)     WHISPER_MODEL="${arg#--whisper=}" ;;
        --summarize=*)   SUMMARIZE=true; OLLAMA_MODEL="${arg#--summarize=}" ;;
        --summarize)     SUMMARIZE=true ;;
        --force-whisper)        FORCE_WHISPER=true ;;
        --force-local-summary)  FORCE_LOCAL_SUMMARY=true ;;
        --help|-h)       usage ;;
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
                exit 1
            fi
            ;;
    esac
done

if [[ -z "$YT_URL" ]]; then
    echo "Error: YouTube URL is required." >&2
    exit 1
fi

VALID_MODELS="tiny base small medium large"
if ! echo "$VALID_MODELS" | grep -qw "$WHISPER_MODEL"; then
    echo "Error: Invalid Whisper model '${WHISPER_MODEL}'. Choose from: ${VALID_MODELS}" >&2
    exit 1
fi

if echo "$GPU_MODELS" | grep -qw "$WHISPER_MODEL"; then
    WHISPER_DEVICE="cuda"
else
    WHISPER_DEVICE="cpu"
fi

# ─── Dependency checks ────────────────────────────────────────────────────────

check_dep() {
    local bin="$1" hint="$2"
    if ! command -v "$bin" &>/dev/null && [[ ! -x "$bin" ]]; then
        echo "Error: '${bin}' not found. ${hint}" >&2
        exit 1
    fi
}

check_dep "$YT_DLP_BIN" "Install with: sudo snap install yt-dlp"
check_dep "ffmpeg"       "Install with: sudo apt install ffmpeg"

if [[ "$SUMMARIZE" == "true" ]]; then
    check_dep "curl" "Install with: sudo apt install curl"
    check_dep "jq"   "Install with: sudo apt install jq"
fi

# Whisper venv only required if we may fall back to it
if [[ "$FORCE_WHISPER" == "true" ]] || [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
    if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
        echo "Error: Python venv not found at ${VENV_PATH}" >&2
        echo "       Create it: python3 -m venv ${VENV_PATH}" >&2
        exit 1
    fi
fi

# ─── Cleanup trap ─────────────────────────────────────────────────────────────

OLLAMA_STARTED=false

cleanup() {
    if [[ "$OLLAMA_STARTED" == "true" ]]; then
        echo ""
        echo "==> Stopping Ollama container (${OLLAMA_CONTAINER})..."
        docker stop "$OLLAMA_CONTAINER" &>/dev/null || true
        docker rm   "$OLLAMA_CONTAINER" &>/dev/null || true
        echo "==> Ollama container removed."
    fi
}
trap cleanup EXIT

# ─── Resolve video metadata ───────────────────────────────────────────────────

echo "==> Resolving video metadata..."
VIDEO_ID=$("$YT_DLP_BIN" --print id "$YT_URL" 2>/dev/null) || {
    echo "Error: Could not resolve video ID. Check the URL or yt-dlp version." >&2
    exit 1
}
VIDEO_TITLE=$("$YT_DLP_BIN" --print title "$YT_URL" 2>/dev/null || echo "unknown_title")
SAFE_TITLE=$(echo "$VIDEO_TITLE" | tr -cd '[:alnum:] _-' | tr ' ' '_' | cut -c1-60)

OUTPUT_DIR="${OUTPUT_BASE}/${VIDEO_ID}"
mkdir -p "$OUTPUT_DIR"

echo "==> Video ID      : ${VIDEO_ID}"
echo "==> Title         : ${VIDEO_TITLE}"
echo "==> Output dir    : ${OUTPUT_DIR}"
echo "==> Whisper model : ${WHISPER_MODEL} (${WHISPER_DEVICE})"
echo "==> Force Whisper : ${FORCE_WHISPER}"
if [[ "$SUMMARIZE" == "true" ]]; then
    if [[ "$FORCE_LOCAL_SUMMARY" == "true" ]]; then
        echo "==> Summarize     : yes (local only: ${OLLAMA_MODEL})"
    elif [[ -n "${OLLAMA_API_KEY:-}" ]]; then
        echo "==> Summarize     : yes (cloud: ${OLLAMA_CLOUD_MODEL} → fallback: ${OLLAMA_MODEL})"
    else
        echo "==> Summarize     : yes (local: ${OLLAMA_MODEL})"
    fi
else
    echo "==> Summarize     : no"
fi
if [[ "$WHISPER_DEVICE" == "cpu" ]]; then
    echo "    Note: medium/large models exceed available VRAM — Whisper falling back to CPU"
fi

# ─── Transcription — subtitle fast path, Whisper fallback ────────────────────
# Strategy:
#   1. Cached transcript     — reuse existing .txt, skip everything
#   2. Human subtitles       — yt-dlp --write-subs, no audio download needed
#   3. Auto-generated subs   — yt-dlp --write-auto-subs, no audio download needed
#   4. Whisper               — download MP3 first, then local inference

echo ""
TRANSCRIPT_FILE=""
TRANSCRIPT_SOURCE=""
AUDIO_FILE=""

# ── Check for cached transcript from a previous run ──────────────────────────
EXISTING_TXT=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*.txt" | head -1)
if [[ -n "$EXISTING_TXT" ]]; then
    echo "==> Transcript already exists, skipping transcription."
    TRANSCRIPT_FILE="$EXISTING_TXT"
    TRANSCRIPT_SOURCE="cached"

elif [[ "$FORCE_WHISPER" == "false" ]]; then

    # ── Attempt 1: human-written subtitles ───────────────────────────────────
    echo "==> Checking for human-written subtitles..."
    "$YT_DLP_BIN" \
        --skip-download \
        --write-subs \
        --sub-lang en \
        --sub-format vtt \
        --output "${OUTPUT_DIR}/%(title)s.%(ext)s" \
        "$YT_URL" 2>/dev/null || true

    VTT_FILE=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*.en.vtt" ! -name "*.live_chat*" | head -1)

    if [[ -n "$VTT_FILE" ]]; then
        echo "==> Human subtitles found: ${VTT_FILE}"
        TRANSCRIPT_FILE="${OUTPUT_DIR}/${SAFE_TITLE}.txt"
        _vtt_to_txt "$VTT_FILE" "$TRANSCRIPT_FILE"
        rm -f "$VTT_FILE"
        TRANSCRIPT_SOURCE="youtube-subtitles"

    else
        # ── Attempt 2: auto-generated subtitles ──────────────────────────────
        echo "==> No human subtitles. Checking for auto-generated subtitles..."
        "$YT_DLP_BIN" \
            --skip-download \
            --write-auto-subs \
            --sub-lang en \
            --sub-format vtt \
            --output "${OUTPUT_DIR}/%(title)s.%(ext)s" \
            "$YT_URL" 2>/dev/null || true

        VTT_FILE=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*.en.vtt" ! -name "*.live_chat*" | head -1)

        if [[ -n "$VTT_FILE" ]]; then
            echo "==> Auto-generated subtitles found: ${VTT_FILE}"
            TRANSCRIPT_FILE="${OUTPUT_DIR}/${SAFE_TITLE}.txt"
            _vtt_to_txt "$VTT_FILE" "$TRANSCRIPT_FILE"
            rm -f "$VTT_FILE"
            TRANSCRIPT_SOURCE="youtube-auto-subtitles"
        fi
    fi
fi

# ── Attempt 3: Whisper local inference (fallback or --force-whisper) ─────────
if [[ -z "$TRANSCRIPT_FILE" ]]; then
    if [[ "$FORCE_WHISPER" == "true" ]]; then
        echo "==> --force-whisper set — using Whisper for transcription."
    else
        echo "==> No YouTube subtitles available — falling back to Whisper."
    fi

    # Audio download only required for Whisper
    EXISTING_MP3=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*.mp3" | head -1)
    if [[ -n "$EXISTING_MP3" ]]; then
        echo "==> Audio already exists, skipping download."
        AUDIO_FILE="$EXISTING_MP3"
    else
        echo "==> Downloading audio for Whisper..."
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

    # shellcheck disable=SC1091
    source "${VENV_PATH}/bin/activate"
    echo "==> Transcribing with Whisper '${WHISPER_MODEL}' on ${WHISPER_DEVICE}..."
    whisper "$AUDIO_FILE" \
        --model "$WHISPER_MODEL" \
        --device "$WHISPER_DEVICE" \
        --output_dir "$OUTPUT_DIR" \
        --output_format txt \
        --verbose False
    deactivate

    TRANSCRIPT_FILE=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*.txt" | head -1)
    TRANSCRIPT_SOURCE="whisper-${WHISPER_MODEL}"
fi

if [[ -z "$TRANSCRIPT_FILE" ]]; then
    echo "Error: All transcription methods failed." >&2
    exit 1
fi

echo "==> Transcript  : ${TRANSCRIPT_FILE}"
echo "==> Source      : ${TRANSCRIPT_SOURCE}"

# ─── Summarize with Ollama (optional) ────────────────────────────────────────

SUMMARY_FILE=""

if [[ "$SUMMARIZE" == "true" ]]; then

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

    # ── Attempt 1: Ollama cloud API ──────────────────────────────────────────
    SUMMARIZE_SOURCE=""
    RESPONSE=""

    if [[ "$FORCE_LOCAL_SUMMARY" != "true" && -n "${OLLAMA_API_KEY:-}" ]]; then
        echo ""
        echo "==> Attempting cloud summarization with '${OLLAMA_CLOUD_MODEL}'..."
        CLOUD_RESPONSE=$(curl -sf -X POST "${OLLAMA_CLOUD_URL}/generate" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer ${OLLAMA_API_KEY}" \
            -d "$(jq -n --arg model "$OLLAMA_CLOUD_MODEL" --arg prompt "$PROMPT" \
                '{model: $model, prompt: $prompt, stream: false, think: false}')" \
            2>&1) || true

        if echo "$CLOUD_RESPONSE" | jq -e '.response' &>/dev/null; then
            RESPONSE="$CLOUD_RESPONSE"
            SUMMARIZE_SOURCE="ollama-cloud:${OLLAMA_CLOUD_MODEL}"
            echo "==> Cloud summarization succeeded."
        else
            echo "==> Cloud summarization failed — reason: $(echo "$CLOUD_RESPONSE" | head -c 200)"
            echo "==> Falling back to local model '${OLLAMA_MODEL}'."
        fi
    else
        echo ""
        if [[ "$FORCE_LOCAL_SUMMARY" == "true" ]]; then
            echo "==> --force-local-summary set — skipping cloud, using local model '${OLLAMA_MODEL}'."
        else
            echo "==> OLLAMA_API_KEY not set — using local model '${OLLAMA_MODEL}'."
        fi
    fi

    # ── Attempt 2: local Ollama container (fallback or no API key) ───────────
    if [[ -z "$RESPONSE" ]]; then
        if [[ -n "$OLLAMA_EXTERNAL" && "$OLLAMA_EXTERNAL" != "http://localhost:"* ]]; then
            echo ""
            echo "==> Using external Ollama service at ${OLLAMA_URL}"
            READY=false
            for i in $(seq 1 30); do
                if curl -sf "${OLLAMA_URL}/api/tags" &>/dev/null; then
                    READY=true; break
                fi
                sleep 1
            done
            [[ "$READY" != "true" ]] && { echo "Error: Ollama not reachable at ${OLLAMA_URL}" >&2; exit 1; }
        else
            echo ""
            echo "==> Starting local Ollama container (${OLLAMA_CONTAINER})..."
            docker volume create "$OLLAMA_VOLUME" &>/dev/null || true
            docker run -d \
                --name "$OLLAMA_CONTAINER" \
                --gpus all \
                -p "${OLLAMA_HOST_PORT}:11434" \
                -v "${OLLAMA_VOLUME}:/root/.ollama" \
                "$OLLAMA_IMAGE" &>/dev/null
            OLLAMA_STARTED=true

            echo "==> Waiting for Ollama API..."
            READY=false
            for i in $(seq 1 30); do
                if curl -sf "${OLLAMA_URL}/api/tags" &>/dev/null; then
                    READY=true; break
                fi
                sleep 1
            done
            [[ "$READY" != "true" ]] && { echo "Error: Ollama did not become ready." >&2; exit 1; }
        fi

        echo "==> Ollama ready. Checking model '${OLLAMA_MODEL}'..."
        MODEL_EXISTS=$(curl -sf "${OLLAMA_URL}/api/tags" | grep -c "\"${OLLAMA_MODEL}\"" || true)
        if [[ "$MODEL_EXISTS" -eq 0 ]]; then
            echo "==> Pulling '${OLLAMA_MODEL}'..."
            curl -sf -X POST "${OLLAMA_URL}/api/pull" \
                -H "Content-Type: application/json" \
                -d "{\"name\": \"${OLLAMA_MODEL}\"}" | grep -v '^$' | tail -1
            echo ""
        else
            echo "==> Model '${OLLAMA_MODEL}' already cached."
        fi

        echo "==> Summarizing with local model '${OLLAMA_MODEL}'..."
        RESPONSE=$(curl -sf -X POST "${OLLAMA_URL}/api/generate" \
            -H "Content-Type: application/json" \
            -d "$(jq -n --arg model "$OLLAMA_MODEL" --arg prompt "$PROMPT" \
                '{model: $model, prompt: $prompt, stream: false, think: false}')")
        SUMMARIZE_SOURCE="ollama-local:${OLLAMA_MODEL}"
    fi

    {
        echo "# ${VIDEO_TITLE}"
        echo ""
        echo "_Source: ${YT_URL}_"
        echo ""
        echo "_Transcribed via: ${TRANSCRIPT_SOURCE} — Summarized with Ollama \`${SUMMARIZE_SOURCE}\`_"
        echo ""
        echo "---"
        echo ""
        echo "$RESPONSE" | jq -r '.response' \
            | perl -0pe 's|<think>.*?</think>\s*||gs'
    } > "$SUMMARY_FILE"

    echo "==> Summary     : ${SUMMARY_FILE}"
fi

# ─── Final report ─────────────────────────────────────────────────────────────

SCRIPT_END=$(date +%s)
ELAPSED=$(( SCRIPT_END - SCRIPT_START ))
ELAPSED_MIN=$(( ELAPSED / 60 ))
ELAPSED_SEC=$(( ELAPSED % 60 ))

echo ""
echo "==> Done."
[[ -n "$AUDIO_FILE" ]] && echo "    Audio            : ${AUDIO_FILE}" || echo "    Audio            : not downloaded (subtitles used)"
echo "    Transcript       : ${TRANSCRIPT_FILE}"
echo "    Transcript source: ${TRANSCRIPT_SOURCE}"
[[ -n "$SUMMARY_FILE" ]] && echo "    Summary          : ${SUMMARY_FILE}"
echo "    All outputs      : ${OUTPUT_DIR}"
echo "    Elapsed time     : ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
