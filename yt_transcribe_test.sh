#!/usr/bin/env bash
# yt_transcribe_test.sh — Test driver for yt_transcribe.sh
#
# Runs a matrix of Whisper (base, small) x Summarize (off, on) combinations
# against a fixed reference video, then copies outputs to a dedicated test
# results directory for side-by-side inspection.
#
# Test matrix (4 cases):
#   1. 1_base_no_summary         base,  no summarize
#   2. 2_base_summarize_gemma3_1b  base,  --summarize=gemma3:1b
#   3. 3_small_no_summary        small, no summarize
#   4. 4_small_summarize_gemma3_1b small, --summarize=gemma3:1b
#
# Reference video:
#   "Carney LOCKS IN Quebec — 173-Seat Majority Just Got More Dangerous for Trump"
#   https://www.youtube.com/watch?v=1F4hNaWsic0
#
# Output layout:
#   ~/Downloads/yt_transcribe_tests/
#     run_YYYYMMDD_HHMMSS/
#       1_base_no_summary/
#         <title>-base.txt
#       2_base_summarize_gemma3_1b/
#         <title>-base.txt
#         <title>_summary-base-gemma3_1b.md
#       3_small_no_summary/
#         <title>-small.txt
#       4_small_summarize_gemma3_1b/
#         <title>-small.txt
#         <title>_summary-small-gemma3_1b.md
#       test_report.txt           Timing and pass/fail summary
#
# Artifact naming convention:
#   <file_name>-<whisper_model>.<ext>             (transcripts)
#   <file_name>-<whisper_model>-<ollama_model>.md (summaries)
#
# Usage:
#   ./yt_transcribe_test.sh
#
# Notes:
#   - Estimated runtime: 10-15 minutes (GPU inference only; CPU models excluded)
#   - MP3 download is cached after the first test case
#   - Each test case writes outputs to a temp staging dir to avoid cross-case
#     contamination before copying to the numbered results directory
#   - Diff transcripts with:
#       diff run_X/1_base_no_summary/<title>-base.txt \
#            run_X/3_small_no_summary/<title>-small.txt
#
# Change history:
#   See git log for revision history

set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRANSCRIBE_SCRIPT="${SCRIPT_DIR}/yt_transcribe.sh"
TEST_URL="https://www.youtube.com/watch?v=1F4hNaWsic0"
VIDEO_ID="1F4hNaWsic0"
DOWNLOADS_BASE="${HOME}/Downloads"
RESULTS_BASE="${HOME}/Downloads/yt_transcribe_tests"
RUN_DIR="${RESULTS_BASE}/run_$(date +%Y%m%d_%H%M%S)"
REPORT_FILE="${RUN_DIR}/test_report.txt"

# Staging directory — isolated per test case to prevent cross-contamination
STAGING_BASE="${DOWNLOADS_BASE}/.yt_transcribe_test_staging"

# ─── Test matrix definition ───────────────────────────────────────────────────
# Format: "run_num|case_dir_suffix|whisper_model|ollama_model"
# ollama_model is empty string when summarize is disabled

TEST_CASES=(
    "1|base_no_summary|base|"
    "2|base_summarize_gemma3_1b|base|gemma3:1b"
    "3|small_no_summary|small|"
    "4|small_summarize_gemma3_1b|small|gemma3:1b"
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

for test_case in "${TEST_CASES[@]}"; do

    # Parse test case fields
    RUN_NUM=$(echo "$test_case"      | cut -d'|' -f1)
    CASE_SUFFIX=$(echo "$test_case"  | cut -d'|' -f2)
    WHISPER=$(echo "$test_case"      | cut -d'|' -f3)
    OLLAMA=$(echo "$test_case"       | cut -d'|' -f4)

    CASE_DIR="${RUN_NUM}_${CASE_SUFFIX}"
    CASE_START=$(date +%s)

    # Per-case isolated staging directory
    STAGING_DIR="${STAGING_BASE}/${VIDEO_ID}_${CASE_DIR}"
    rm -rf "$STAGING_DIR"
    mkdir -p "$STAGING_DIR"

    log ""
    log "==> [${RUN_NUM}/${TOTAL}] ${CASE_DIR}"
    log "    Whisper  : ${WHISPER}"
    log "    Ollama   : ${OLLAMA:-none}"
    log "    Staging  : ${STAGING_DIR}"
    log "    Started  : $(date '+%H:%M:%S')"

    # Build argument list — point OUTPUT_BASE at isolated staging dir by
    # temporarily overriding via env; yt_transcribe.sh uses HOME-relative path
    # so we pass the URL with a staging-aware workaround using a wrapper call
    ARGS=("$TEST_URL" "--whisper=${WHISPER}")
    if [[ -n "$OLLAMA" ]]; then
        ARGS+=("--summarize=${OLLAMA}")
    fi

    # Run transcribe script with OUTPUT_BASE overridden via sed-patched env
    # We achieve staging isolation by symlinking the staging dir as the video ID
    # subfolder inside a temp output root, then passing that root via a patched
    # copy of the script's OUTPUT_BASE at runtime using an env wrapper.
    TEMP_OUTPUT_ROOT="${STAGING_BASE}/output_${CASE_DIR}"
    rm -rf "$TEMP_OUTPUT_ROOT"
    mkdir -p "$TEMP_OUTPUT_ROOT"

    set +e
    OUTPUT_BASE="$TEMP_OUTPUT_ROOT" \
    bash -c "
        source_script='${TRANSCRIBE_SCRIPT}'
        # Re-run with OUTPUT_BASE overridden inside the script environment
        sed 's|OUTPUT_BASE=\"\${HOME}/Downloads\"|OUTPUT_BASE=\"${TEMP_OUTPUT_ROOT}\"|' \
            \"\$source_script\" | bash -s -- ${ARGS[*]}
    " 2>&1 | sed "s/^/    [${CASE_DIR}] /" | tee -a "$REPORT_FILE"
    EXIT_CODE=${PIPESTATUS[0]}
    set -e

    CASE_END=$(date +%s)
    CASE_ELAPSED=$(( CASE_END - CASE_START ))
    CASE_MIN=$(( CASE_ELAPSED / 60 ))
    CASE_SEC=$(( CASE_ELAPSED % 60 ))

    # ── Copy and rename artifacts to results directory ────────────────────────

    DEST_DIR="${RUN_DIR}/${CASE_DIR}"
    mkdir -p "$DEST_DIR"

    VIDEO_OUT_DIR="${TEMP_OUTPUT_ROOT}/${VIDEO_ID}"
    COPIED=0

    # Sanitize model names for use in filenames (replace : with _)
    WHISPER_SAFE="${WHISPER}"
    OLLAMA_SAFE=$(echo "$OLLAMA" | tr ':' '_')

    if [[ -d "$VIDEO_OUT_DIR" ]]; then

        # Copy transcript: <name>.txt → <name>-<whisper>.txt
        TXT_FILE=$(find "$VIDEO_OUT_DIR" -maxdepth 1 -name "*.txt" | head -1)
        if [[ -n "$TXT_FILE" ]]; then
            BASENAME=$(basename "$TXT_FILE" .txt)
            cp "$TXT_FILE" "${DEST_DIR}/${BASENAME}-${WHISPER_SAFE}.txt"
            COPIED=$((COPIED + 1))
        fi

        # Copy summary: <name>_summary.md → <name>_summary-<whisper>-<ollama>.md
        MD_FILE=$(find "$VIDEO_OUT_DIR" -maxdepth 1 -name "*_summary.md" | head -1)
        if [[ -n "$MD_FILE" ]]; then
            BASENAME=$(basename "$MD_FILE" .md)
            # Strip trailing _summary suffix to rebuild cleanly
            BASE_NOSUM="${BASENAME%_summary}"
            cp "$MD_FILE" "${DEST_DIR}/${BASE_NOSUM}_summary-${WHISPER_SAFE}-${OLLAMA_SAFE}.md"
            COPIED=$((COPIED + 1))
        fi

    fi

    # Clean up staging output for this case
    rm -rf "$TEMP_OUTPUT_ROOT"

    if [[ $EXIT_CODE -eq 0 ]]; then
        log "    Result   : PASS (${CASE_MIN}m ${CASE_SEC}s, ${COPIED} file(s) copied)"
        PASS=$((PASS + 1))
    else
        log "    Result   : FAIL (exit code ${EXIT_CODE}, ${CASE_MIN}m ${CASE_SEC}s)"
        FAIL=$((FAIL + 1))
    fi

    log_separator
done

# Clean up staging base
rm -rf "$STAGING_BASE"

# ─── Final report ─────────────────────────────────────────────────────────────

OVERALL_END=$(date +%s)
OVERALL_ELAPSED=$(( OVERALL_END - OVERALL_START ))
OVERALL_MIN=$(( OVERALL_ELAPSED / 60 ))
OVERALL_SEC=$(( OVERALL_ELAPSED % 60 ))

log ""
log "Test run complete."
log "  Finished  : $(date '+%Y-%m-%d %H:%M:%S')"
log "  Elapsed   : ${OVERALL_MIN}m ${OVERALL_SEC}s"
log "  Passed    : ${PASS}/${TOTAL}"
log "  Failed    : ${FAIL}/${TOTAL}"
log ""
log "Results saved to: ${RUN_DIR}"
log ""
log "Suggested diffs:"
log ""
log "  Transcripts (base vs small):"
log "    diff '${RUN_DIR}/1_base_no_summary/'*-base.txt \\"
log "         '${RUN_DIR}/3_small_no_summary/'*-small.txt"
log ""
log "  Summaries (base vs small):"
log "    diff '${RUN_DIR}/2_base_summarize_gemma3_1b/'*-base-gemma3_1b.md \\"
log "         '${RUN_DIR}/4_small_summarize_gemma3_1b/'*-small-gemma3_1b.md"
log_separator

exit $FAIL
