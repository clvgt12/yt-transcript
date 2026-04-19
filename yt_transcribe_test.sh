#!/usr/bin/env bash
# yt_transcribe_test.sh — Test driver for yt_transcribe.sh
#
# Test matrix (6 cases):
#   1. subtitles_only                    YouTube subtitles, no summarize
#   2. subtitles_summarize_gemma3        YouTube subtitles, --summarize=gemma3:1b
#   3. force_whisper_base_no_summary     --force-whisper --whisper=base, no summarize
#   4. force_whisper_base_summarize_gemma3 --force-whisper --whisper=base, --summarize=gemma3:1b
#   5. force_whisper_small_no_summary    --force-whisper --whisper=small, no summarize
#   6. subtitles_summarize_qwen3         YouTube subtitles, --summarize=qwen3:1.7b
#
# Qualitative alignment analysis:
#   After all cases complete, the report automatically diffs:
#     - Subtitles vs Whisper small
#     - Subtitles vs Whisper base
#     - Whisper small vs Whisper base
#   Results are written to <run_dir>/diff_analysis/ for side-by-side inspection.
#
# Reference video:
#   "Carney LOCKS IN Quebec — 173-Seat Majority Just Got More Dangerous for Trump"
#   https://www.youtube.com/watch?v=1F4hNaWsic0
#
# Output layout:
#   ~/Downloads/yt_transcribe_tests/
#     run_YYYYMMDD_HHMMSS/
#       1_subtitles_only/
#         <title>-subtitles.txt
#       2_subtitles_summarize_gemma3/
#         <title>-subtitles.txt
#         <title>_summary-subtitles-gemma3_1b.md
#       3_force_whisper_base_no_summary/
#         <title>-whisper_base.txt
#       4_force_whisper_base_summarize_gemma3/
#         <title>-whisper_base.txt
#         <title>_summary-whisper_base-gemma3_1b.md
#       5_force_whisper_small_no_summary/
#         <title>-whisper_small.txt
#       6_subtitles_summarize_qwen3/
#         <title>-subtitles.txt
#         <title>_summary-subtitles-qwen3_1.7b.md
#       diff_analysis/
#         diff_subtitles_vs_whisper_small.txt
#         diff_subtitles_vs_whisper_base.txt
#         diff_whisper_small_vs_whisper_base.txt
#         alignment_report.txt
#       test_report.txt
#
# Usage:
#   ./yt_transcribe_test.sh
#
# Notes:
#   - Cases 1,2,6 use YouTube subtitle fast path — near instant transcription
#   - Cases 3,4,5 use --force-whisper — GPU inference, ~3-5 min each
#   - Each case runs in an isolated staging directory to prevent contamination
#   - Estimated total runtime: 15-20 minutes
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
DIFF_DIR="${RUN_DIR}/diff_analysis"
STAGING_BASE="${DOWNLOADS_BASE}/.yt_transcribe_test_staging"

# ─── Test matrix ──────────────────────────────────────────────────────────────
# Format: "run_num|case_suffix|extra_args|whisper_label|ollama_label"
# extra_args    — passed directly to yt_transcribe.sh (space-separated)
# whisper_label — short label used in output filename (subtitles / whisper_small / etc.)
# ollama_label  — ollama model name sanitized, empty if no summarize

TEST_CASES=(
    "1|subtitles_only|--whisper=small|subtitles|"
    "2|subtitles_summarize_gemma3|--whisper=small --summarize=gemma3:1b|subtitles|gemma3_1b"
    "3|force_whisper_base_no_summary|--whisper=base --force-whisper|whisper_base|"
    "4|force_whisper_base_summarize_gemma3|--whisper=base --force-whisper --summarize=gemma3:1b|whisper_base|gemma3_1b"
    "5|force_whisper_small_no_summary|--whisper=small --force-whisper|whisper_small|"
    "6|subtitles_summarize_qwen3|--whisper=small --summarize=qwen3:1.7b|subtitles|qwen3_1.7b"
)

TOTAL=${#TEST_CASES[@]}

# ─── Preflight checks ─────────────────────────────────────────────────────────

if [[ ! -x "$TRANSCRIBE_SCRIPT" ]]; then
    echo "Error: yt_transcribe.sh not found or not executable at ${TRANSCRIBE_SCRIPT}" >&2
    exit 1
fi

mkdir -p "$RUN_DIR" "$DIFF_DIR"

# ─── Logging helpers ──────────────────────────────────────────────────────────

log() { echo "$*" | tee -a "$REPORT_FILE"; }

log_separator() {
    log "──────────────────────────────────────────────────────────────────"
}

# ─── Report header ────────────────────────────────────────────────────────────

log_separator
log "yt_transcribe.sh — Test Run"
log "Started  : $(date '+%Y-%m-%d %H:%M:%S')"
log "Run dir  : ${RUN_DIR}"
log "Video    : ${TEST_URL}"
log "Cases    : ${TOTAL}"
log_separator

PASS=0
FAIL=0
OVERALL_START=$(date +%s)

# Track artifact paths for diff analysis
declare -A TRANSCRIPT_PATHS

# ─── Run test matrix ──────────────────────────────────────────────────────────

for test_case in "${TEST_CASES[@]}"; do

    RUN_NUM=$(echo "$test_case"       | cut -d'|' -f1)
    CASE_SUFFIX=$(echo "$test_case"   | cut -d'|' -f2)
    EXTRA_ARGS=$(echo "$test_case"    | cut -d'|' -f3)
    WHISPER_LABEL=$(echo "$test_case" | cut -d'|' -f4)
    OLLAMA_LABEL=$(echo "$test_case"  | cut -d'|' -f5)

    CASE_DIR="${RUN_NUM}_${CASE_SUFFIX}"
    CASE_START=$(date +%s)

    # Isolated staging directory for this case
    TEMP_OUTPUT_ROOT="${STAGING_BASE}/output_${CASE_DIR}"
    rm -rf "$TEMP_OUTPUT_ROOT"
    mkdir -p "$TEMP_OUTPUT_ROOT"

    log ""
    log "==> [${RUN_NUM}/${TOTAL}] ${CASE_DIR}"
    log "    Args     : ${EXTRA_ARGS}"
    log "    Labels   : whisper=${WHISPER_LABEL} ollama=${OLLAMA_LABEL:-none}"
    log "    Started  : $(date '+%H:%M:%S')"

    # Build args array
    read -ra ARGS <<< "$EXTRA_ARGS"

    # Run with OUTPUT_BASE overridden via sed-patched script
    set +e
    sed "s|OUTPUT_BASE=\"\${OUTPUT_BASE:-\${HOME}/Downloads}\"|OUTPUT_BASE=\"${TEMP_OUTPUT_ROOT}\"|" \
        "$TRANSCRIBE_SCRIPT" | bash -s -- "$TEST_URL" "${ARGS[@]}" \
        2>&1 | sed "s/^/    [${CASE_DIR}] /" | tee -a "$REPORT_FILE"
    EXIT_CODE=${PIPESTATUS[0]}
    set -e

    CASE_END=$(date +%s)
    CASE_ELAPSED=$(( CASE_END - CASE_START ))
    CASE_MIN=$(( CASE_ELAPSED / 60 ))
    CASE_SEC=$(( CASE_ELAPSED % 60 ))

    # ── Copy and rename artifacts ─────────────────────────────────────────────

    DEST_DIR="${RUN_DIR}/${CASE_DIR}"
    mkdir -p "$DEST_DIR"

    VIDEO_OUT_DIR="${TEMP_OUTPUT_ROOT}/${VIDEO_ID}"
    COPIED=0
    SAVED_TRANSCRIPT=""

    if [[ -d "$VIDEO_OUT_DIR" ]]; then

        # Transcript: <name>.txt → <name>-<whisper_label>.txt
        TXT_FILE=$(find "$VIDEO_OUT_DIR" -maxdepth 1 -name "*.txt" | head -1)
        if [[ -n "$TXT_FILE" ]]; then
            BASENAME=$(basename "$TXT_FILE" .txt)
            DEST_TXT="${DEST_DIR}/${BASENAME}-${WHISPER_LABEL}.txt"
            cp "$TXT_FILE" "$DEST_TXT"
            SAVED_TRANSCRIPT="$DEST_TXT"
            COPIED=$((COPIED + 1))
        fi

        # Summary: <name>_summary.md → <name>_summary-<whisper_label>-<ollama_label>.md
        MD_FILE=$(find "$VIDEO_OUT_DIR" -maxdepth 1 -name "*_summary.md" | head -1)
        if [[ -n "$MD_FILE" && -n "$OLLAMA_LABEL" ]]; then
            BASENAME=$(basename "$MD_FILE" .md)
            BASE_NOSUM="${BASENAME%_summary}"
            cp "$MD_FILE" "${DEST_DIR}/${BASE_NOSUM}_summary-${WHISPER_LABEL}-${OLLAMA_LABEL}.md"
            COPIED=$((COPIED + 1))
        fi
    fi

    # Store transcript path for diff analysis keyed by label
    if [[ -n "$SAVED_TRANSCRIPT" ]]; then
        TRANSCRIPT_PATHS["$WHISPER_LABEL_${RUN_NUM}"]="$SAVED_TRANSCRIPT"
        # Also store by canonical label for diff lookup
        case "$CASE_SUFFIX" in
            subtitles_only)                    TRANSCRIPT_PATHS["subtitles"]="$SAVED_TRANSCRIPT" ;;
            force_whisper_base_no_summary)     TRANSCRIPT_PATHS["whisper_base"]="$SAVED_TRANSCRIPT" ;;
            force_whisper_small_no_summary)    TRANSCRIPT_PATHS["whisper_small"]="$SAVED_TRANSCRIPT" ;;
        esac
    fi

    # Clean up staging
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

rm -rf "$STAGING_BASE"

# ─── Diff analysis ────────────────────────────────────────────────────────────

log ""
log "==> Running transcript alignment analysis..."

ALIGNMENT_REPORT="${DIFF_DIR}/alignment_report.txt"

{
    echo "Transcript Alignment Analysis"
    echo "Run: ${RUN_DIR}"
    echo "Generated: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "======================================================"
} > "$ALIGNMENT_REPORT"

run_diff() {
    local label_a="$1"
    local label_b="$2"
    local file_a="${TRANSCRIPT_PATHS[$label_a]:-}"
    local file_b="${TRANSCRIPT_PATHS[$label_b]:-}"
    local diff_out="${DIFF_DIR}/diff_${label_a}_vs_${label_b}.txt"

    if [[ -z "$file_a" || ! -f "$file_a" ]]; then
        log "    SKIP diff ${label_a} vs ${label_b} — ${label_a} transcript not found"
        return
    fi
    if [[ -z "$file_b" || ! -f "$file_b" ]]; then
        log "    SKIP diff ${label_a} vs ${label_b} — ${label_b} transcript not found"
        return
    fi

    # Word counts
    WORDS_A=$(wc -w < "$file_a")
    WORDS_B=$(wc -w < "$file_b")
    LINES_A=$(wc -l < "$file_a")
    LINES_B=$(wc -l < "$file_b")

    # Diff stats
    DIFF_LINES=$(diff "$file_a" "$file_b" | grep -c '^[<>]' || true)

    {
        echo ""
        echo "------------------------------------------------------"
        echo "Comparison: ${label_a}  vs  ${label_b}"
        echo "------------------------------------------------------"
        echo "  File A (${label_a}): ${file_a}"
        echo "    Words: ${WORDS_A}  Lines: ${LINES_A}"
        echo "  File B (${label_b}): ${file_b}"
        echo "    Words: ${WORDS_B}  Lines: ${LINES_B}"
        echo "  Changed lines (diff): ${DIFF_LINES}"
        echo ""
        echo "  Full diff → $(basename "$diff_out")"
    } >> "$ALIGNMENT_REPORT"

    # Write full unified diff
    {
        echo "Diff: ${label_a} vs ${label_b}"
        echo "A: ${file_a}"
        echo "B: ${file_b}"
        echo "======================================================"
        diff --unified=2 "$file_a" "$file_b" || true
    } > "$diff_out"

    log "    Diff ${label_a} vs ${label_b}: ${DIFF_LINES} changed lines → $(basename "$diff_out")"
}

run_diff "subtitles"     "whisper_base"
run_diff "subtitles"     "whisper_small"
run_diff "whisper_base"  "whisper_small"

log ""
log "    Alignment report → ${ALIGNMENT_REPORT}"

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
log "Diff analysis:"
log "  ${DIFF_DIR}/alignment_report.txt"
log "  ${DIFF_DIR}/diff_subtitles_vs_whisper_base.txt"
log "  ${DIFF_DIR}/diff_subtitles_vs_whisper_small.txt"
log "  ${DIFF_DIR}/diff_whisper_base_vs_whisper_small.txt"
log_separator

exit $FAIL
