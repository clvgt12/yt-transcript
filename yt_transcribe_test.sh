#!/usr/bin/env bash
# yt_transcribe_test.sh — Test driver for yt_transcribe.sh
#
# Runs a matrix of Whisper (base, small) x Summarize (off, on) combinations
# against a fixed reference video, then copies outputs to a dedicated test
# results directory for side-by-side inspection.
#
# Test matrix (4 cases):
#   1. base,  no summarize
#   2. base,  --summarize=gemma3:1b
#   3. small, no summarize
#   4. small, --summarize=gemma3:1b
#
# Reference video:
#   "Carney LOCKS IN Quebec — 173-Seat Majority Just Got More Dangerous for Trump"
#   https://www.youtube.com/watch?v=1F4hNaWsic0
#
# Output layout:
#   ~/Downloads/yt_transcribe_tests/
#     run_YYYYMMDD_HHMMSS/
#       base_no_summary/
#         <title>.txt
#       base_summarize_gemma3_1b/
#         <title>.txt
#         <title>.md
#       small_no_summary/
#         <title>.txt
#       small_summarize_gemma3_1b/
#         <title>.txt
#         <title>.md
#       test_report.txt       Timing and pass/fail summary
#
# Usage:
#   ./yt_transcribe_test.sh
#
# Notes:
#   - Estimated runtime: 10-15 minutes (GPU inference only; CPU models excluded)
#   - MP3 download is cached after the first test case
#   - Diff transcripts with: diff run_A/base_no_summary/<title>.txt \
#                                  run_B/base_no_summary/<title>.txt
#
# Change history:
#   See git log for revision history

set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRANSCRIBE_SCRIPT="${SCRIPT_DIR}/yt_transcribe.sh"
TEST_URL="https://www.youtube.com/watch?v=1F4hNaWsic0"
VIDEO_ID="1F4hNaWsic0"
OLLAMA_MODEL="gemma3:1b"
DOWNLOADS_BASE="${HOME}/Downloads"
RESULTS_BASE="${HOME}/Downloads/yt_transcribe_tests"
RUN_DIR="${RESULTS_BASE}/run_$(date +%Y%m%d_%H%M%S)"
REPORT_FILE="${RUN_DIR}/test_report.txt"

# ─── Test matrix definition ───────────────────────────────────────────────────
# Each entry: "case_dir_name|whisper_model|summarize_flag"
# summarize_flag is empty string for no summarize, or "--summarize=MODEL"

TEST_CASES=(
    "base_no_summary|base|"
    "base_summarize_gemma3_1b|base|--summarize=${OLLAMA_MODEL}"
    "small_no_summary|small|"
    "small_summarize_gemma3_1b|small|--summarize=${OLLAMA_MODEL}"
)

TOTAL=${#TEST_CASES[@]}

# ─── Preflight checks ─────────────────────────────────────────────────────────

if [[ ! -x "$TRANSCRIBE_SCRIPT" ]]; then
    echo "Error: yt_transcribe.sh not found or not executable at ${TRANSCRIBE_SCRIPT}" >&2
    exit 1
fi

mkdir -p "$RUN_DIR"

# ─── Logging helpers ──────────────────────────────────────────────────────────

log() {
    echo "$*" | tee -a "$REPORT_FILE"
}

log_separator() {
    log "──────────────────────────────────────────────────────────────────"
}

# ─── Report header ────────────────────────────────────────────────────────────

log_separator
log "yt_transcribe.sh — Test Run"
log "Started : $(date '+%Y-%m-%d %H:%M:%S')"
log "Run dir : ${RUN_DIR}"
log "Video   : ${TEST_URL}"
log "Cases   : ${TOTAL}"
log_separator

PASS=0
FAIL=0
OVERALL_START=$(date +%s)

# ─── Run test matrix ──────────────────────────────────────────────────────────

CASE_NUM=0
for test_case in "${TEST_CASES[@]}"; do
    CASE_NUM=$((CASE_NUM + 1))

    # Parse test case fields
    CASE_DIR=$(echo "$test_case"  | cut -d'|' -f1)
    WHISPER=$(echo "$test_case"   | cut -d'|' -f2)
    SUMMARIZE=$(echo "$test_case" | cut -d'|' -f3)

    CASE_START=$(date +%s)

    log ""
    log "==> [${CASE_NUM}/${TOTAL}] ${CASE_DIR}"
    log "    Whisper : ${WHISPER}"
    log "    Summarize: ${SUMMARIZE:-none}"
    log "    Started : $(date '+%H:%M:%S')"

    # Build argument list
    ARGS=("$TEST_URL" "--whisper=${WHISPER}")
    if [[ -n "$SUMMARIZE" ]]; then
        ARGS+=("$SUMMARIZE")
    fi

    # Run the transcription script, streaming output with a case prefix
    set +e
    "$TRANSCRIBE_SCRIPT" "${ARGS[@]}" 2>&1 | sed "s/^/    [${CASE_DIR}] /" | tee -a "$REPORT_FILE"
    EXIT_CODE=${PIPESTATUS[0]}
    set -e

    CASE_END=$(date +%s)
    CASE_ELAPSED=$(( CASE_END - CASE_START ))
    CASE_MIN=$(( CASE_ELAPSED / 60 ))
    CASE_SEC=$(( CASE_ELAPSED % 60 ))

    # Copy output files to results directory
    DEST_DIR="${RUN_DIR}/${CASE_DIR}"
    mkdir -p "$DEST_DIR"

    VIDEO_OUT_DIR="${DOWNLOADS_BASE}/${VIDEO_ID}"
    COPIED=0

    if [[ -d "$VIDEO_OUT_DIR" ]]; then
        # Copy transcript
        TXT_FILE=$(find "$VIDEO_OUT_DIR" -maxdepth 1 -name "*.txt" | head -1)
        if [[ -n "$TXT_FILE" ]]; then
            cp "$TXT_FILE" "$DEST_DIR/"
            COPIED=$((COPIED + 1))
        fi
        # Copy summary if present
        MD_FILE=$(find "$VIDEO_OUT_DIR" -maxdepth 1 -name "*_summary.md" | head -1)
        if [[ -n "$MD_FILE" ]]; then
            cp "$MD_FILE" "$DEST_DIR/"
            COPIED=$((COPIED + 1))
        fi
    fi

    if [[ $EXIT_CODE -eq 0 ]]; then
        log "    Result  : PASS (${CASE_MIN}m ${CASE_SEC}s, ${COPIED} file(s) copied)"
        PASS=$((PASS + 1))
    else
        log "    Result  : FAIL (exit code ${EXIT_CODE}, ${CASE_MIN}m ${CASE_SEC}s)"
        FAIL=$((FAIL + 1))
    fi

    log_separator
done

# ─── Final report ─────────────────────────────────────────────────────────────

OVERALL_END=$(date +%s)
OVERALL_ELAPSED=$(( OVERALL_END - OVERALL_START ))
OVERALL_MIN=$(( OVERALL_ELAPSED / 60 ))
OVERALL_SEC=$(( OVERALL_ELAPSED % 60 ))

log ""
log "Test run complete."
log "  Finished : $(date '+%Y-%m-%d %H:%M:%S')"
log "  Elapsed  : ${OVERALL_MIN}m ${OVERALL_SEC}s"
log "  Passed   : ${PASS}/${TOTAL}"
log "  Failed   : ${FAIL}/${TOTAL}"
log ""
log "Results saved to: ${RUN_DIR}"
log ""
log "Diff transcripts between Whisper models:"
log "  diff ${RUN_DIR}/base_no_summary/*.txt \\"
log "       ${RUN_DIR}/small_no_summary/*.txt"
log ""
log "Diff summaries between Whisper models:"
log "  diff ${RUN_DIR}/base_summarize_gemma3_1b/*_summary.md \\"
log "       ${RUN_DIR}/small_summarize_gemma3_1b/*_summary.md"
log_separator

# Mirror exit code — non-zero if any case failed
exit $FAIL
