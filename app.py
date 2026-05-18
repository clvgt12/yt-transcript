"""
yt-transcribe-web — Streamlit web interface for YouTube transcription + summarization

Job state lives entirely in st.session_state so it survives Streamlit reruns
within the same browser session. The background worker thread updates the
job object in-place; the UI reads it on every poll cycle.
"""

import json
import os
import shutil
import re
import sys
import uuid
import logging
import threading
import subprocess
import textwrap
import time
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st
import markdown as md_lib
import requests

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("yt-transcribe-web")

# ─── Configuration from environment ──────────────────────────────────────────

FILES_BASE          = Path(os.environ.get("FILES_BASE",         "/usr/app/files"))
YT_DLP_BIN          = os.environ.get("YT_DLP_BIN",             "/usr/local/bin/yt-dlp")
VENV_PATH           = os.environ.get("VENV_PATH",              "/venv")
WHISPER_MODEL       = os.environ.get("WHISPER_MODEL",          "small")
GPU_MODELS          = os.environ.get("GPU_MODELS",             "tiny base small").split()
SUMMARIZE           = os.environ.get("SUMMARIZE",              "true").lower() == "true"
OLLAMA_MODEL        = os.environ.get("OLLAMA_MODEL",           "qwen3:1.7b")
OLLAMA_URL          = os.environ.get("OLLAMA_URL",             "http://ollama:11434")
OLLAMA_CLOUD_URL    = os.environ.get("OLLAMA_CLOUD_URL",       "https://ollama.com/api")
OLLAMA_CLOUD_MODEL  = os.environ.get("OLLAMA_CLOUD_MODEL",     "gpt-oss:120b")
OLLAMA_API_KEY      = os.environ.get("OLLAMA_API_KEY",         "")
FORCE_LOCAL_SUMMARY = os.environ.get("FORCE_LOCAL_SUMMARY",    "false").lower() == "true"
FORCE_WHISPER       = os.environ.get("FORCE_WHISPER",          "false").lower() == "true"
POLL_INTERVAL_MS    = int(os.environ.get("POLL_INTERVAL_MS",   "2000"))
CACHE_FILE_AGE_DAYS    = int(os.environ.get("CACHE_FILE_AGE_DAYS",    "30"))
WEB_SEARCH_ENABLED     = os.environ.get("WEB_SEARCH_ENABLED",     "true").lower() == "true"
WEB_SEARCH_MAX_RESULTS = int(os.environ.get("WEB_SEARCH_MAX_RESULTS", "5"))

FILES_BASE.mkdir(parents=True, exist_ok=True)

# ─── Web search ───────────────────────────────────────────────────────────────

def web_search(query: str, max_results: int = 5) -> list:
    """
    Search the web using DuckDuckGo. Returns list of result dicts:
    [{title, href, body}, ...]
    Returns empty list if search is disabled or fails.
    """
    if not WEB_SEARCH_ENABLED:
        return []
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url":   r.get("href",  ""),
                    "body":  r.get("body",  ""),
                })
        log.info("Web search '%s': %d results", query, len(results))
        return results
    except Exception as exc:
        log.warning("Web search failed: %s", exc)
        return []


def format_search_results(results: list) -> str:
    """Format search results as a readable context block for the LLM prompt."""
    if not results:
        return ""
    lines = ["--- Web Search Results ---"]
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r['title']}")
        lines.append(f"    URL: {r['url']}")
        lines.append(f"    {r['body']}")
        lines.append("")
    return "\n".join(lines)


# ─── Cache management ─────────────────────────────────────────────────────────

def purge_expired_cache():
    """Remove video output directories older than CACHE_FILE_AGE_DAYS."""
    if CACHE_FILE_AGE_DAYS <= 0:
        return  # 0 = disabled
    cutoff = time.time() - (CACHE_FILE_AGE_DAYS * 86400)
    purged = 0
    for entry in FILES_BASE.iterdir():
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        # Use the most recent mtime among all files in the directory
        mtimes = [f.stat().st_mtime for f in entry.rglob("*") if f.is_file()]
        if not mtimes:
            continue
        newest_mtime = max(mtimes)
        if newest_mtime < cutoff:
            try:
                shutil.rmtree(entry)
                log.info("Cache purged (expired): %s", entry.name)
                purged += 1
            except PermissionError:
                log.warning("Cache purge skipped (permission denied): %s — "
                            "run: sudo chown -R ubuntu:ubuntu %s", entry.name, entry)
    if purged:
        log.info("Cache purge complete: %d director%s removed.", purged, "y" if purged == 1 else "ies")


def find_cached_job(video_id: str) -> dict:
    """
    Check if a completed job exists in cache for this video_id.
    Returns dict with keys: transcript_path, html_path, transcript, summary_html
    or empty dict if no usable cache entry exists.
    """
    out_dir = FILES_BASE / video_id
    if not out_dir.is_dir():
        return {}

    txts  = [f for f in out_dir.glob("*.txt") if f.is_file()]
    htmls = [f for f in out_dir.glob("*.html") if f.is_file()]

    if not txts:
        return {}

    result = {
        "transcript_path": txts[0],
        "transcript":      txts[0].read_text(encoding="utf-8"),
        "html_path":       None,
        "summary_html":    None,
    }

    if htmls:
        html_content = htmls[0].read_text(encoding="utf-8")
        result["html_path"]    = htmls[0]
        result["summary_html"] = html_content

    return result


def save_chat_history(video_id: str, history: list):
    """Persist chat history to <video_id>/chat_history.json."""
    path = FILES_BASE / video_id / "chat_history.json"
    path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def load_chat_history(video_id: str) -> list:
    """Load chat history from disk, return empty list if not found."""
    path = FILES_BASE / video_id / "chat_history.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def chat_with_ollama(history: list, transcript: str, title: str,
                     search_results: list = None) -> Optional[str]:
    """
    Send full conversation history to Ollama (cloud first, local fallback).
    history: list of {role: user|assistant, content: str}
    search_results: optional list of web search results to inject as context
    Returns assistant reply text or None on failure.
    """
    search_context = ""
    if search_results:
        search_context = f"""
The following web search results provide additional real-world context
beyond the transcript. Use them to enrich your answer where relevant,
and cite the source URL when drawing from them.

{format_search_results(search_results)}
"""

    system_msg = textwrap.dedent(f"""
        You are a helpful analyst assistant with access to web search results.
        The user is asking follow-up questions about a YouTube video titled: "{title}"

        IMPORTANT INSTRUCTIONS:
        - Respond directly with your answer. Do NOT narrate your search process,
          show intermediate reasoning steps, or describe what you are about to do.
        - Do NOT write phrases like "Searching...", "Let me search...",
          "Simulated result list:", or "Now answer." — go straight to the answer.
        - Use both the transcript and any web search results provided.
        - Clearly distinguish between information from the transcript and web sources.
        - Cite source URLs when drawing from web search results.
        - If neither source contains the answer, say so clearly and concisely.
        {search_context}
        ---
        TRANSCRIPT:
        {transcript}
    """).strip()

    # Build messages array for /api/chat endpoint
    messages = [{"role": "system", "content": system_msg}] + history

    # Cloud first
    if not FORCE_LOCAL_SUMMARY and OLLAMA_API_KEY:
        try:
            resp = requests.post(
                f"{OLLAMA_CLOUD_URL}/chat",
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {OLLAMA_API_KEY}"},
                json={"model": OLLAMA_CLOUD_MODEL, "messages": messages,
                      "stream": False, "think": False},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            reply = data.get("message", {}).get("content") or data.get("response")
            if reply:
                return strip_think(reply)
        except Exception as exc:
            log.warning("Cloud chat failed: %s — trying local.", exc)

    # Local fallback
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": OLLAMA_MODEL, "messages": messages,
                  "stream": False, "think": False},
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()
        reply = data.get("message", {}).get("content") or data.get("response")
        if reply:
            return strip_think(reply)
    except Exception as exc:
        log.error("Local chat failed: %s", exc)

    return None


# ─── Job class ────────────────────────────────────────────────────────────────

class Job:
    """
    Mutable job object stored directly in st.session_state["job"].
    Background thread updates it in-place; UI reads it on every poll rerun.
    Thread-safe via a simple Lock on log appends.
    """
    def __init__(self, job_id: str, url: str):
        self.job_id        = job_id
        self.url           = url
        self.status        = "queued"   # queued | running | done | failed
        self.log_lines     = []
        self.summary_html: Optional[str] = None
        self.error: Optional[str] = None
        self.started_at    = datetime.utcnow().isoformat()
        self.finished_at: Optional[str] = None
        self.video_id: Optional[str] = None
        self.video_title: Optional[str] = None
        self.transcript: Optional[str] = None
        self.chat_history: list = []      # [{role, content}, ...]
        self._lock         = threading.Lock()

    def log(self, line: str):
        with self._lock:
            self.log_lines.append(line)
        log.info("[%s] %s", self.job_id[:8], line)

    def get_log(self) -> list:
        with self._lock:
            return list(self.log_lines)


# ─── Workflow helpers ─────────────────────────────────────────────────────────

def extract_video_id(url: str) -> Optional[str]:
    parsed = urllib.parse.urlparse(url)
    if parsed.hostname in ("youtu.be",):
        return parsed.path.lstrip("/").split("?")[0]
    qs = urllib.parse.parse_qs(parsed.query)
    return qs.get("v", [None])[0]


def sanitize_title(title: str) -> str:
    safe = re.sub(r"[^\w\s\-]", "", title)
    safe = re.sub(r"\s+", "_", safe.strip())
    return safe[:80]


def vtt_to_text(vtt_path: Path) -> str:
    raw = vtt_path.read_text(encoding="utf-8")
    raw = re.sub(r"^WEBVTT[^\n]*\n.*?\n\n", "", raw, count=1, flags=re.DOTALL)
    blocks = re.split(r"\n{2,}", raw)
    sentences, para_breaks = [], set()
    prev_line, prev_sent_end = None, False
    for block in blocks:
        if not block.strip():
            continue
        lines = block.strip().splitlines()
        if lines and re.match(r"^\s*\d+\s*$", lines[0]):
            lines = lines[1:]
        if lines and re.match(r"\d{2}:\d{2}[\d:,.]+\s*-->\s*\d{2}:\d{2}", lines[0]):
            lines = lines[1:]
        if lines and lines[0].startswith("NOTE"):
            continue
        cleaned = []
        for line in lines:
            line = re.sub(r"<[^>]+>", "", line)
            line = re.sub(r"&amp;", "&", line).replace("&lt;","<").replace("&gt;",">").strip()
            if line:
                cleaned.append(line)
        for line in cleaned:
            if line == prev_line:
                continue
            sentences.append(line)
            if prev_sent_end:
                para_breaks.add(len(sentences) - 1)
            prev_line = line
            prev_sent_end = bool(re.search(r"[.!?]\s*$", line))
    merged = []
    for i, sent in enumerate(sentences):
        if (merged and not re.search(r"[.!?,;:]\s*$", merged[-1])
                and i not in para_breaks and sent[:1].islower()):
            merged[-1] = merged[-1].rstrip() + " " + sent
        else:
            merged.append(sent)
    paragraphs, current = [], []
    for i, sent in enumerate(merged):
        current.append(sent)
        if (re.search(r"[.!?]\s*$", sent)
                and i + 1 < len(merged) and merged[i + 1][:1].isupper()
                and (i + 1) in para_breaks):
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)


def build_prompt(transcript: str, title: str = "") -> str:
    title_section = ""
    if title:
        title_section = f"""
## Title Alignment
Evaluate how well the transcript content aligns with the video title: "{title}"
Answer the following:
- **Alignment verdict**: Choose one: Strongly aligned | Partially aligned | Weakly aligned | Misleading
- **Explanation**: 1-3 sentences explaining your verdict. Cite specific evidence from the transcript.
- **What the transcript actually covers**: One sentence describing the real subject matter if it differs from the title.

"""

    return textwrap.dedent(f"""
        You are a professional analyst. Read the following transcript carefully and produce
        a structured summary in Markdown format with exactly {"four" if title else "three"} sections:

        ## Summary
        Write a concise summary of 3-5 sentences covering the core subject and conclusions.

        ## Key Points
        Bullet list of the most important facts, arguments, or events from the transcript.

        ## Takeaways
        Bullet list of the key insights, implications, or action items a reader should walk away with.
        {title_section}
        Use clean Markdown formatting. Be precise and objective. Do not editorialize.

        ---
        TRANSCRIPT:
        {transcript}
    """).strip()


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def markdown_to_html(md_text: str, title: str, url: str, source: str, model: str) -> str:
    body = md_lib.markdown(md_text, extensions=["extra", "nl2br"])
    return textwrap.dedent(f"""
        <!DOCTYPE html><html lang="en"><head>
        <meta charset="UTF-8"><title>{title}</title>
        <style>
          body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,
               Helvetica,Arial,sans-serif;max-width:860px;margin:2rem auto;
               padding:0 1.5rem;line-height:1.7;color:#222}}
          h1{{font-family:Georgia,'Times New Roman',serif;font-size:1.6rem;
              border-bottom:2px solid #ccc;padding-bottom:.4rem}}
          h2{{font-family:Georgia,'Times New Roman',serif;font-size:1.2rem;
              margin-top:2rem;color:#333}}
          ul{{padding-left:1.4rem}} li{{margin-bottom:.4rem}}
          .meta{{font-size:.85rem;color:#666;margin-bottom:1.5rem}}
          hr{{border:none;border-top:1px solid #ddd;margin:1.5rem 0}}
        </style></head><body>
        <h1>{title}</h1>
        <div class="meta">
          <a href="{url}" target="_blank">{url}</a><br>
          Transcribed via: {source} &mdash; Summarized with: {model}
        </div><hr>{body}
        </body></html>
    """).strip()


# ─── Background workflow ──────────────────────────────────────────────────────

def run_workflow(job: Job):
    """Full pipeline — mutates job object in place, read by UI via session_state."""
    job.status = "running"
    t_start = time.time()

    try:
        job.log("Resolving video metadata...")
        video_id = extract_video_id(job.url)
        if not video_id:
            r = subprocess.run([YT_DLP_BIN, "--print", "id", job.url],
                               capture_output=True, text=True)
            video_id = r.stdout.strip()
        if not video_id:
            raise RuntimeError("Could not extract video ID.")

        r2 = subprocess.run([YT_DLP_BIN, "--print", "title", job.url],
                            capture_output=True, text=True)
        video_title = r2.stdout.strip() or "unknown_title"
        safe_title  = sanitize_title(video_title)

        job.log(f"Video ID : {video_id}")
        job.log(f"Title    : {video_title}")
        job.video_id    = video_id
        job.video_title = video_title

        out_dir = FILES_BASE / video_id
        out_dir.mkdir(parents=True, exist_ok=True)
        txt_path  = out_dir / f"{safe_title}.txt"
        html_path = out_dir / f"{safe_title}.html"

        # ── Cache hit check ───────────────────────────────────────────────────
        cached = find_cached_job(video_id)
        if cached and cached.get("summary_html") and not FORCE_WHISPER:
            job.log("Cache hit — reusing existing transcript and summary.")
            job.log(f"Transcript : {cached['transcript_path'].name}")
            job.log(f"Summary    : {cached['html_path'].name}")
            job.video_id     = video_id
            job.video_title  = video_title
            job.summary_html = cached["summary_html"]
            job.transcript   = cached["transcript"]
            job.chat_history = load_chat_history(video_id)
            if job.chat_history:
                job.log(f"Chat history restored: {len(job.chat_history)} messages.")
            elapsed = int(time.time() - t_start)
            job.log(f"Done (from cache). Elapsed: {elapsed // 60}m {elapsed % 60}s")
            job.status      = "done"
            job.finished_at = datetime.utcnow().isoformat()
            return

        # ── Transcription ─────────────────────────────────────────────────────
        transcript_source = "cached"

        if txt_path.exists() and not FORCE_WHISPER:
            job.log(f"Transcript cached: {txt_path.name}")
            transcript = txt_path.read_text(encoding="utf-8")
        elif not FORCE_WHISPER:
            transcript = None
            for sub_flag, label in [("--write-subs", "human"),
                                     ("--write-auto-subs", "auto-generated")]:
                job.log(f"Checking for {label} subtitles...")
                subprocess.run([
                    YT_DLP_BIN, "--skip-download", sub_flag,
                    "--sub-lang", "en", "--sub-format", "vtt",
                    "--output", str(out_dir / "%(title)s.%(ext)s"), job.url,
                ], capture_output=True)
                vtts = [f for f in out_dir.glob("*.en.vtt") if "live_chat" not in f.name]
                if vtts:
                    job.log(f"Found {label} subtitles.")
                    transcript = vtt_to_text(vtts[0])
                    vtts[0].unlink(missing_ok=True)
                    txt_path.write_text(transcript, encoding="utf-8")
                    transcript_source = f"youtube-{label.replace(' ', '-')}"
                    break
        else:
            transcript = None

        if transcript is None:
            job.log("Using Whisper for transcription...")
            mp3s = list(out_dir.glob("*.mp3"))
            if not mp3s:
                job.log("Downloading audio...")
                subprocess.run([
                    YT_DLP_BIN, "--extract-audio", "--audio-format", "mp3",
                    "--audio-quality", "0",
                    "--output", str(out_dir / "%(title)s.%(ext)s"), job.url,
                ], capture_output=True)
                mp3s = list(out_dir.glob("*.mp3"))
            if not mp3s:
                raise RuntimeError("Audio download failed.")
            audio_path = mp3s[0]
            device = "cuda" if WHISPER_MODEL in GPU_MODELS else "cpu"
            job.log(f"Transcribing with Whisper '{WHISPER_MODEL}' on {device}...")
            activate = Path(VENV_PATH) / "bin" / "activate"
            cmd = (f"source {activate} && whisper '{audio_path}' "
                   f"--model {WHISPER_MODEL} --device {device} "
                   f"--output_dir '{out_dir}' --output_format txt --verbose False")
            subprocess.run(cmd, shell=True, executable="/bin/bash", capture_output=True)
            txts = list(out_dir.glob("*.txt"))
            if not txts:
                raise RuntimeError("Whisper produced no output.")
            transcript = txts[0].read_text(encoding="utf-8")
            txt_path = txts[0]
            transcript_source = f"whisper-{WHISPER_MODEL}"

        job.log(f"Transcript source: {transcript_source}")
        job.transcript = transcript

        # ── Summarization ─────────────────────────────────────────────────────
        summary_md    = None
        summary_model = "none"

        if SUMMARIZE:
            prompt = build_prompt(transcript, title=video_title)

            # Cloud first
            if not FORCE_LOCAL_SUMMARY and OLLAMA_API_KEY:
                job.log(f"Attempting cloud summarization with '{OLLAMA_CLOUD_MODEL}'...")
                try:
                    resp = requests.post(
                        f"{OLLAMA_CLOUD_URL}/generate",
                        headers={"Content-Type": "application/json",
                                 "Authorization": f"Bearer {OLLAMA_API_KEY}"},
                        json={"model": OLLAMA_CLOUD_MODEL, "prompt": prompt,
                              "stream": False, "think": False},
                        timeout=120,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    if "response" in data:
                        summary_md    = strip_think(data["response"])
                        summary_model = f"ollama-cloud:{OLLAMA_CLOUD_MODEL}"
                        job.log("Cloud summarization succeeded.")
                except Exception as exc:
                    job.log(f"Cloud failed ({exc}) — trying local.")

            # Local fallback
            if summary_md is None:
                job.log(f"Summarizing with local model '{OLLAMA_MODEL}'...")
                for attempt in range(30):
                    try:
                        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
                        if r.status_code == 200:
                            break
                    except Exception:
                        pass
                    job.log(f"Waiting for Ollama API... ({attempt + 1}/30)")
                    time.sleep(2)
                try:
                    resp = requests.post(
                        f"{OLLAMA_URL}/api/generate",
                        json={"model": OLLAMA_MODEL, "prompt": prompt,
                              "stream": False, "think": False},
                        timeout=300,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    if "response" in data:
                        summary_md    = strip_think(data["response"])
                        summary_model = f"ollama-local:{OLLAMA_MODEL}"
                except Exception as exc:
                    job.log(f"Local summarization failed: {exc}")

        # ── Write output files ────────────────────────────────────────────────
        if summary_md:
            html_content = markdown_to_html(
                summary_md, video_title, job.url, transcript_source, summary_model)
            html_path.write_text(html_content, encoding="utf-8")
            job.summary_html = html_content
            job.log(f"Summary saved: {html_path.name}")
            job.chat_history = load_chat_history(video_id)   # restore any prior chat
        else:
            job.log("No summary produced.")

        elapsed = int(time.time() - t_start)
        job.log(f"Done. Elapsed: {elapsed // 60}m {elapsed % 60}s")
        job.status      = "done"
        job.finished_at = datetime.utcnow().isoformat()

    except Exception as exc:
        job.log(f"ERROR: {exc}")
        job.error       = str(exc)
        job.status      = "failed"
        job.finished_at = datetime.utcnow().isoformat()
        log.exception("Workflow failed for job %s", job.job_id)


# ─── Streamlit UI ─────────────────────────────────────────────────────────────

def render_css():
    st.markdown("""<style>
    .block-container{max-width:860px;padding-top:2rem}
    .input-div{background:#f8f9fa;border:1px solid #dee2e6;border-radius:8px;
               padding:1.5rem 2rem;margin-bottom:1.5rem}
    .output-div{background:#fff;border:1px solid #dee2e6;border-radius:8px;
                padding:1.5rem 2rem;min-height:120px}
    .log-box{background:#0a0a0a;color:#00ff00;font-family:'Courier New',monospace;
             font-size:.82rem;line-height:1.5;padding:1rem;border-radius:6px;
             max-height:320px;overflow-y:auto;white-space:pre-wrap;
             font-weight:bold;text-shadow:0 0 5px rgba(0,255,0,0.5)}
    </style>""", unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="YT Transcribe", page_icon="🎬", layout="centered")
    render_css()
    st.title("🎬 YouTube Transcribe & Summarize")

    # ── Cache purge on session startup (once per session) ───────────────────────
    if "cache_purged" not in st.session_state:
        purge_expired_cache()
        st.session_state.cache_purged = True

    # ── Initialise session state ───────────────────────────────────────────────
    if "job" not in st.session_state:
        st.session_state.job = None

    # ── INPUT DIV ─────────────────────────────────────────────────────────────
    st.markdown('<div class="input-div">', unsafe_allow_html=True)
    st.markdown("#### Input")
    # Counter-based key forces widget re-instantiation on clear, resetting its value
    if "input_counter" not in st.session_state:
        st.session_state.input_counter = 0

    with st.form("transcribe_form", clear_on_submit=False):
        yt_url = st.text_input(
            "YouTube URL",
            key=f"yt_url_input_{st.session_state.input_counter}",
            placeholder="https://www.youtube.com/watch?v=...",
        )
        col_submit, col_clear = st.columns([3, 1])
        with col_submit:
            submitted = st.form_submit_button("▶  Submit", use_container_width=True)
        with col_clear:
            cleared = st.form_submit_button("✕  Clear", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if cleared:
        st.session_state.input_counter += 1  # new key → new widget instance → empty value
        st.session_state.job = None
        st.rerun()

    # Handle submission — create Job, store in session_state, start thread
    if submitted and yt_url.strip():
        url = yt_url.strip()
        if "youtube.com" not in url and "youtu.be" not in url:
            st.error("Please enter a valid YouTube URL.")
        else:
            job = Job(str(uuid.uuid4()), url)
            st.session_state.job = job          # store object, not just id
            t = threading.Thread(
                target=run_workflow, args=(job,),
                daemon=True, name=f"worker-{job.job_id[:8]}"
            )
            t.start()
            log.info("Job started: %s — %s", job.job_id, url)
            st.rerun()

    # ── OUTPUT DIV ────────────────────────────────────────────────────────────
    st.markdown('<div class="output-div">', unsafe_allow_html=True)
    st.markdown("#### Output")

    job: Optional[Job] = st.session_state.get("job")

    if job is None:
        st.markdown(
            '<p style="color:#888;font-style:italic;">Submit a YouTube URL above to begin.</p>',
            unsafe_allow_html=True)

    elif job.status == "done":
        if job.summary_html:
            st.components.v1.html(job.summary_html, height=600, scrolling=True)
        else:
            st.success("✅ Transcription complete. No summary was produced.")

        # ── Chat follow-up ─────────────────────────────────────────────────────
        if job.transcript:
            st.markdown("---")
            st.markdown("#### 💬 Ask a follow-up question")

            # Render existing chat history
            if job.chat_history:
                for msg in job.chat_history:
                    role  = msg["role"]
                    label = "**You:**" if role == "user" else "**Assistant:**"
                    bg    = "#f0f4ff" if role == "user" else "#f8f9fa"
                    st.markdown(
                        f'<div style="background:{bg};border-radius:6px;'
                        f'padding:.6rem 1rem;margin:.4rem 0;">'
                        f'{label}<br>{msg["content"]}</div>',
                        unsafe_allow_html=True,
                    )

            # Chat input form
            if "chat_counter" not in st.session_state:
                st.session_state.chat_counter = 0

            with st.form(f"chat_form_{st.session_state.chat_counter}"):
                user_q = st.text_area(
                    "Your question",
                    placeholder="Ask anything about this video...",
                    height=80,
                    label_visibility="collapsed",
                )
                chat_submitted = st.form_submit_button("Send ➤", use_container_width=False)

            if chat_submitted and user_q.strip():
                with st.spinner("Searching and thinking..."):
                    # Run web search on the user question for additional context
                    search_results = web_search(
                        user_q.strip(), max_results=WEB_SEARCH_MAX_RESULTS
                    )
                    if search_results:
                        log.info("Injecting %d web results into chat context",
                                 len(search_results))

                    job.chat_history.append({"role": "user", "content": user_q.strip()})
                    reply = chat_with_ollama(
                        job.chat_history, job.transcript,
                        job.video_title or "this video",
                        search_results=search_results,
                    )
                    if reply:
                        job.chat_history.append({"role": "assistant", "content": reply})
                        if job.video_id:
                            save_chat_history(job.video_id, job.chat_history)
                    else:
                        job.chat_history.pop()   # remove unanswered user message
                        st.error("Chat request failed. Please try again.")
                st.session_state.chat_counter += 1
                st.rerun()

        if st.button("🔄 Transcribe another video"):
            st.session_state.input_counter += 1  # new key → new widget instance → empty value
            st.session_state.job = None
            st.rerun()

    elif job.status == "failed":
        st.error(f"❌ Job failed: {job.error or 'Unknown error'}")
        lines = job.get_log()
        if lines:
            st.markdown(
                f'<div class="log-box">{"<br>".join(lines)}</div>',
                unsafe_allow_html=True)
        if st.button("🔄 Try again"):
            st.session_state.job = None
            st.rerun()

    else:
        # Queued or running — poll
        label = "⏳ Running..." if job.status == "running" else "📋 Queued"
        st.markdown(f'<p style="color:#0d6efd;font-weight:600">{label}</p>',
                    unsafe_allow_html=True)
        lines = job.get_log()
        if lines:
            st.markdown(
                f'<div class="log-box">{"<br>".join(lines)}</div>',
                unsafe_allow_html=True)
        time.sleep(POLL_INTERVAL_MS / 1000)
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(
        "<hr><p style='text-align:center;color:#aaa;font-size:.8rem;'>"
        "yt-transcribe-web &mdash; kamakazi</p>",
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()
