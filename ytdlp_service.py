"""
ytdlp-service — FastAPI microservice for YouTube metadata, subtitle and audio download.

API:
    POST /download
        Start a download job. Deduplicates by video_id — if a job for the same
        video_id is already running, returns the existing job_id immediately.
        Parameters (all except video_id are optional):
          video_id : str   — YouTube video ID (mandatory)
          meta     : bool  — fetch title and available format info
          human    : bool  — download human-written subtitles → transcript
          auto     : bool  — download auto-generated subtitles → transcript
          audio    : bool  — download mp3 audio file

    GET /download/{job_id}
        Poll job status and retrieve results. Resets per-job watchdog timer.

    DELETE /download/{job_id}
        Cancel a running job, kill subprocess, clean up partial files.

    GET /health
        Health check.

Per-job watchdog: resets on each GET poll, fires if client stops polling.
"""

import os
import re
import sys
import uuid
import json
import logging
import threading
import subprocess
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("ytdlp-service")

# ─── Configuration ─────────────────────────────────────────────────────────────

YT_DLP_BIN          = os.environ.get("YT_DLP_BIN",          "/usr/local/bin/yt-dlp")
FILES_BASE          = Path(os.environ.get("FILES_BASE",      "/files"))
JOB_TIMEOUT_SECONDS = int(os.environ.get("JOB_TIMEOUT_SECONDS", "300"))  # 5 min
_POLL_INTERVAL_MS   = int(os.environ.get("POLL_INTERVAL_MS",    "2000"))
POLL_TIMEOUT_SECONDS = (3 * _POLL_INTERVAL_MS // 1000) + 1

# Subtitle language selection.
#   Human subs: manual captions are not affected by YouTube's timedtext 429
#   throttling. skip=translated_subs keeps yt-dlp from choosing a translated
#   manual track.
#   Auto subs: request ONLY the native ASR track ("en-orig"). Since late
#   Sep 2026, plain "en" on many videos (e.g. auto-dubbed ones) resolves to a
#   machine translation (timedtext ...&tlang=en) of another ASR track, which
#   YouTube answers with HTTP 429. Non-English videos have no en-orig and
#   fall back to Whisper (see whisper-service TARGET_LANG).
HUMAN_SUB_LANGS = ["en", "en-US", "en-GB"]
AUTO_SUB_LANGS  = ["en-orig"]

FILES_BASE.mkdir(parents=True, exist_ok=True)

# Mitigate HTTP 429 errors and YouTube bot detection
#   Implement bgutil-provider sidecar within ytdlp container

BGUTIL_BASE_URL = os.environ.get("BGUTIL_BASE_URL", "http://bgutil-provider:4416")
POT_ARGS = ["--extractor-args", f"youtubepot-bgutilhttp:base_url={BGUTIL_BASE_URL}"]

# ─── Extraction cache: one YouTube extraction per video ──────────────────────
INFO_DIR         = Path(os.environ.get("INFO_DIR", "/tmp/ytdlp-info"))
INFO_TTL_SECONDS = int(os.environ.get("INFO_TTL_SECONDS", "600"))   # 10 min
VIDEO_ID_RE      = re.compile(r"[A-Za-z0-9_-]{11}")
INFO_DIR.mkdir(parents=True, exist_ok=True)
try:
    os.chmod(INFO_DIR, 0o700)   # info.json holds signed, IP-bound stream URLs
except OSError:
    pass

# ─── Per-job watchdog ─────────────────────────────────────────────────────────

class JobWatchdog:
    """
    Per-job countdown timer. Reset on each GET poll. Fires when client
    disconnects or stops polling, killing the active subprocess.
    """
    def __init__(self, job_id: str, poll_timeout: int, job_timeout: int,
                 on_timeout):
        self.job_id       = job_id
        self.poll_timeout = poll_timeout
        self.job_timeout  = job_timeout
        self.on_timeout   = on_timeout
        self._remaining   = poll_timeout
        self._lock        = threading.Lock()
        self._stopped     = threading.Event()
        self._thread      = threading.Thread(
            target=self._run, daemon=True,
            name=f"watchdog-{job_id[:8]}"
        )

    def start(self):
        self._thread.start()
        log.info("[%s] Watchdog started (poll_timeout=%ds, job_timeout=%ds)",
                 self.job_id[:8], self.poll_timeout, self.job_timeout)

    def reset(self):
        with self._lock:
            self._remaining = self.poll_timeout

    def stop(self):
        self._stopped.set()

    def _run(self):
        job_start  = time.time()
        tick       = max(1, _POLL_INTERVAL_MS // 1000)
        while not self._stopped.is_set():
            time.sleep(tick)
            if time.time() - job_start >= self.job_timeout:
                log.warning("[%s] Watchdog: hard job timeout", self.job_id[:8])
                self.on_timeout("job timeout")
                return
            with self._lock:
                self._remaining -= tick
                remaining = self._remaining
            if remaining <= 0:
                log.warning("[%s] Watchdog: client disconnected", self.job_id[:8])
                self.on_timeout("client disconnected")
                return
        log.info("[%s] Watchdog stopped", self.job_id[:8])


# ─── Job model ────────────────────────────────────────────────────────────────

class Job:
    def __init__(self, job_id: str, video_id: str,
                 meta: bool, human: bool, auto: bool, audio: bool):
        self.job_id     = job_id
        self.video_id   = video_id
        self.meta       = meta
        self.human      = human
        self.auto       = auto
        self.audio      = audio
        self.status     = "queued"   # queued | running | done | failed
        # Results
        self.title:            Optional[str] = None
        self.channel:          Optional[str] = None   # YouTube channel/uploader name
        self.subtitle_source:  Optional[str] = None   # human | auto | none
        self.transcript:       Optional[str] = None
        self.audio_path:       Optional[str] = None
        self.error:            Optional[str] = None
        # Lifecycle
        self.created_at  = datetime.now(UTC).isoformat()
        self.finished_at: Optional[str] = None
        self.process:    Optional[subprocess.Popen] = None
        self.watchdog:   Optional[JobWatchdog]      = None

# ─── Job store — keyed by video_id for deduplication ─────────────────────────

_jobs_by_id:       dict[str, Job] = {}   # job_id  → Job
_jobs_by_video_id: dict[str, str] = {}   # video_id → job_id (active jobs only)
_store_lock = threading.Lock()

def get_job(job_id: str) -> Optional[Job]:
    with _store_lock:
        return _jobs_by_id.get(job_id)

def get_active_job_for_video(video_id: str) -> Optional[Job]:
    with _store_lock:
        job_id = _jobs_by_video_id.get(video_id)
        return _jobs_by_id.get(job_id) if job_id else None

def store_job(job: Job):
    with _store_lock:
        _jobs_by_id[job.job_id]           = job
        _jobs_by_video_id[job.video_id]   = job.job_id

def release_video_id(video_id: str):
    """Remove video_id reservation so future requests start a new job."""
    with _store_lock:
        _jobs_by_video_id.pop(video_id, None)

# ─── Kill / cleanup helper ────────────────────────────────────────────────────

def kill_job(job: Job, reason: str):
    """Kill active subprocess, clean up partial files, mark job failed.
    No-op if job already completed successfully — preserves downloaded files."""
    if job.status == "done":
        # Job already completed — watchdog fired in race window, nothing to do
        log.info("[%s] kill_job called on completed job — ignoring (%s)",
                 job.job_id[:8], reason)
        if job.watchdog:
            job.watchdog.stop()
        return

    proc = job.process
    if proc is not None:
        try:
            if proc.returncode is None:
                proc.kill()
                log.info("[%s] Process killed (%s)", job.job_id[:8], reason)
        except Exception as exc:
            log.warning("[%s] Kill error: %s", job.job_id[:8], exc)
    job.process     = None
    job.status      = "failed"
    job.error       = f"Job terminated: {reason}"
    job.finished_at = datetime.now(UTC).isoformat()
    if job.watchdog:
        job.watchdog.stop()
    # Remove all media files — job did not complete successfully
    _cleanup_media(job.video_id)
    release_video_id(job.video_id)

def _cleanup_media(video_id: str):
    """
    Force-remove all media files for a video_id.
    Used on explicit client cancellation (DELETE) — unconditional cleanup.
    Preserves .txt transcript and .html summary files.
    """
    out_dir = FILES_BASE / video_id
    if not out_dir.is_dir():
        return
    removed = []
    for pattern in ("*.mp3", "*.webm", "*.m4a", "*.part", "*.ytdl", "*.vtt"):
        for f in out_dir.glob(pattern):
            try:
                f.unlink()
                removed.append(f.name)
            except Exception:
                pass
    if removed:
        log.info("[%s] Force-cleaned media files: %s", video_id, ", ".join(removed))

# ─── VTT → plain text converter ──────────────────────────────────────────────

def vtt_to_text(vtt_path: Path) -> str:
    """Convert VTT subtitle file to clean paragraph text."""
    import re as _re
    raw = vtt_path.read_text(encoding="utf-8")
    raw = _re.sub(r"^WEBVTT[^\n]*\n.*?\n\n", "", raw, count=1, flags=_re.DOTALL)
    blocks = _re.split(r"\n{2,}", raw)
    sentences, para_breaks = [], set()
    prev_line, prev_sent_end = None, False
    for block in blocks:
        if not block.strip():
            continue
        lines = block.strip().splitlines()
        if lines and _re.match(r"^\s*\d+\s*$", lines[0]):
            lines = lines[1:]
        if lines and _re.match(r"\d{2}:\d{2}[\d:,.]+\s*-->\s*\d{2}:\d{2}", lines[0]):
            lines = lines[1:]
        if lines and lines[0].startswith("NOTE"):
            continue
        cleaned = []
        for line in lines:
            line = _re.sub(r"<[^>]+>", "", line)
            line = _re.sub(r"&amp;", "&", line).replace("&lt;","<").replace("&gt;",">").strip()
            if line:
                cleaned.append(line)
        for line in cleaned:
            if line == prev_line:
                continue
            sentences.append(line)
            if prev_sent_end:
                para_breaks.add(len(sentences) - 1)
            prev_line = line
            prev_sent_end = bool(_re.search(r"[.!?]\s*$", line))
    merged = []
    for i, sent in enumerate(sentences):
        if (merged and not _re.search(r"[.!?,;:]\s*$", merged[-1])
                and i not in para_breaks and sent[:1].islower()):
            merged[-1] = merged[-1].rstrip() + " " + sent
        else:
            merged.append(sent)
    paragraphs, current = [], []
    for i, sent in enumerate(merged):
        current.append(sent)
        if (_re.search(r"[.!?]\s*$", sent)
                and i + 1 < len(merged) and merged[i + 1][:1].isupper()
                and (i + 1) in para_breaks):
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)


# ─── Download worker ──────────────────────────────────────────────────────────

def _run_cmd(job: Job, cmd: list) -> tuple[int, str, str]:
    """Run a subprocess, store handle on job for cancellation, return rc/stdout/stderr."""
    # Inject the PO Token provider URL into every yt-dlp invocation.
    if cmd and cmd[0] == YT_DLP_BIN:
        cmd = [cmd[0], *POT_ARGS, *cmd[1:]]
    if job.status == "failed":
        # Cancelled between steps: don't start work nobody is waiting for.
        return -1, "", "cancelled"
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    job.process = proc   # cancel/watchdog may set job.process to None at any time
    if job.status == "failed":
        proc.kill()      # cancelled while we were launching
    stdout, stderr = proc.communicate()
    job.process = None
    return proc.returncode, stdout.strip(), stderr.strip()

def _info_path(video_id: str) -> Path:
    return INFO_DIR / f"{video_id}.info.json"

def _read_cached_info(video_id: str) -> Optional[dict]:
    """Saved extraction if fresh and parseable, else None."""
    p = _info_path(video_id)
    try:
        if time.time() - p.stat().st_mtime > INFO_TTL_SECONDS:
            return None
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

def _purge_old_info(max_age: int = 3600):
    cutoff = time.time() - max_age
    for f in INFO_DIR.glob("*.info.json"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
        except OSError:
            pass

def _extract_info(job: Job) -> Optional[dict]:
    """
    The only step that asks YouTube about a video (watch page + player API).
    Returns None if the job was cancelled; raises RuntimeError on failure.
    """
    video_id = job.video_id
    info = _read_cached_info(video_id)
    if info is not None:
        log.info("[%s] Reusing saved extraction for %s", job.job_id[:8], video_id)
        return info

    _purge_old_info()
    log.info("[%s] Extracting %s from YouTube", job.job_id[:8], video_id)
    p = _info_path(video_id)
    p.unlink(missing_ok=True)
    rc, _, stderr = _run_cmd(job, [
        YT_DLP_BIN, "--skip-download", "--no-playlist", "--write-info-json",
        "--output", f"infojson:{INFO_DIR}/%(id)s",
        "--", f"https://www.youtube.com/watch?v={video_id}",
    ])
    if job.status == "failed":
        return None

    js_warning = "no supported javascript runtime" in stderr.lower()
    if js_warning:
        log.warning("[%s] yt-dlp JS runtime warning: %s", job.job_id[:8], stderr[:120])
    if rc == 0:
        try:
            info = json.loads(p.read_text(encoding="utf-8"))
            os.chmod(p, 0o600)
            return info
        except (OSError, ValueError):
            pass
    log.warning("[%s] extraction failed rc=%s file_written=%s; stderr tail: %s",
                job.job_id[:8], rc, p.exists(),
                " | ".join(stderr.splitlines()[-8:])[:1200])
    if js_warning:
        raise RuntimeError(
            "YouTube requires a JavaScript runtime for this video. "
            "Deno is being installed — please retry after container rebuild."
        )
    raise RuntimeError(f"Metadata fetch failed: {_yt_error(stderr)}")

def _pick_sub_lang(info: dict, *, auto: bool) -> Optional[str]:
    """First wanted language with a real (non-translated) track, else None."""
    tracks = (info.get("automatic_captions") if auto else info.get("subtitles")) or {}
    for lang in (AUTO_SUB_LANGS if auto else HUMAN_SUB_LANGS):
        entries = tracks.get(lang) or []
        # tlang= marks a machine translation, the case behind the 429s (#13831)
        if entries and not any("tlang=" in (e.get("url") or "") for e in entries):
            return lang
    return None

def _fetch_subs(job: Job, out_dir: Path, info: dict, info_path: Path,
                *, auto: bool) -> Optional[Path]:
    """
    Download one English subtitle track from the saved extraction; return the
    .vtt path or None. Makes no request when no suitable track exists. Never
    raises for a missing/blocked track -- it logs why so the caller can fall
    back to Whisper. Returns None if cancelled; callers must check job.status.
    """
    video_id = job.video_id
    kind     = "auto" if auto else "human"
    wanted   = AUTO_SUB_LANGS if auto else HUMAN_SUB_LANGS

    lang = _pick_sub_lang(info, auto=auto)
    if lang is None:
        log.info("[%s] No %s subtitles available (%s)",
                 job.job_id[:8], kind, ",".join(wanted))
        return None

    for stale in out_dir.glob(f"{video_id}.*.vtt"):
        stale.unlink(missing_ok=True)

    rc, _, stderr = _run_cmd(job, [
        YT_DLP_BIN, "--load-info-json", str(info_path), "--skip-download",
        "--write-auto-subs" if auto else "--write-subs",
        "--sub-langs", lang, "--sub-format", "vtt",
        "--output", str(out_dir / "%(id)s.%(ext)s"),
    ])
    if job.status == "failed":
        return None

    p = out_dir / f"{video_id}.{lang}.vtt"
    if p.exists() and p.stat().st_size > 0:
        return p

    if "429" in stderr:
        log.warning("[%s] %s subtitles rate-limited by YouTube (HTTP 429)",
                    job.job_id[:8], kind)
    elif rc != 0:
        tail = " | ".join(stderr.strip().splitlines()[-3:])
        log.warning("[%s] %s subtitle fetch failed rc=%s: %s",
                    job.job_id[:8], kind, rc, tail[:300])
    else:
        log.info("[%s] %s subtitles not written (%s)", job.job_id[:8], kind, lang)
    return None

def _yt_error(stderr: str, limit: int = 400) -> str:
    """Prefer yt-dlp's ERROR lines; warnings alone hide the real cause."""
    lines = stderr.splitlines()
    errs = [l for l in lines if l.startswith("ERROR")]
    text = " | ".join(errs) if errs else " | ".join(lines[-3:])
    if "not a bot" in text:
        return ("YouTube bot check (often transient; wait a few minutes before "
                "retrying). Full details are in the ytdlp-service log.")
    return text[:limit]

def download_worker(job: Job):
    """Execute requested download operations sequentially."""
    job.status = "running"
    video_id   = job.video_id
    out_dir    = FILES_BASE / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("[%s] Starting download worker (meta=%s human=%s auto=%s audio=%s)",
             job.job_id[:8], job.meta, job.human, job.auto, job.audio)

    try:
        # ── 0. One extraction, shared by every step below ─────────────────────
        info = _extract_info(job)
        if job.status == "failed":
            return
        info_path = _info_path(video_id)

        # ── 1. Metadata (read from the saved extraction, no network) ──────────
        if job.meta:
            job.title   = (info.get("title") or video_id).strip()
            job.channel = (info.get("channel") or info.get("uploader") or "").strip() or None
            log.info("[%s] Title: %s | Channel: %s",
                     job.job_id[:8], job.title, job.channel)

        # ── 2. Human subtitles ────────────────────────────────────────────────
        if job.human:
            vtt = _fetch_subs(job, out_dir, info, info_path, auto=False)
            if job.status == "failed":
                return
            if vtt:
                job.transcript      = vtt_to_text(vtt)
                job.subtitle_source = "human"
                vtt.unlink(missing_ok=True)
                log.info("[%s] Human subtitles downloaded (%s)",
                         job.job_id[:8], vtt.name)

        # ── 3. Auto-generated subtitles (native ASR track only) ───────────────
        if job.auto and not job.transcript:
            vtt = _fetch_subs(job, out_dir, info, info_path, auto=True)
            if job.status == "failed":
                return
            if vtt:
                job.transcript      = vtt_to_text(vtt)
                job.subtitle_source = "auto"
                vtt.unlink(missing_ok=True)
                log.info("[%s] Auto subtitles downloaded (%s)",
                         job.job_id[:8], vtt.name)
            else:
                job.subtitle_source = "none"

        # ── 4. Audio (mp3), from the saved extraction ─────────────────────────
        if job.audio:
            log.info("[%s] Downloading audio...", job.job_id[:8])
            audio_args = ["--extract-audio", "--audio-format", "mp3", "--audio-quality", "0",
                          "--output", str(out_dir / "%(title)s.%(ext)s")]
            rc, _, stderr = _run_cmd(job, [YT_DLP_BIN, "--load-info-json", str(info_path), *audio_args])
            if job.status == "failed":
                return
            if rc != 0 and "HTTP Error 403" in stderr:
                # Saved stream URLs were rejected: drop them so no retry can reuse
                # them, and make one direct attempt (fresh extraction + download).
                log.warning("[%s] saved extraction rejected (403); retrying once directly",
                            job.job_id[:8])
                _info_path(video_id).unlink(missing_ok=True)
                rc, _, stderr = _run_cmd(job, [YT_DLP_BIN, *audio_args, "--",
                                               f"https://www.youtube.com/watch?v={video_id}"])
                if job.status == "failed":
                    return
            if rc != 0:
                log.warning("[%s] audio download failed rc=%s; stderr tail: %s", job.job_id[:8], rc,
                            " | ".join(stderr.splitlines()[-8:])[:1200])
                raise RuntimeError(f"Audio download failed: {_yt_error(stderr)}")
            mp3s = list(out_dir.glob("*.mp3"))
            if not mp3s:
                raise RuntimeError("Audio download produced no mp3")
            job.audio_path = str(mp3s[0])
            log.info("[%s] Audio saved: %s", job.job_id[:8], mp3s[0].name)

        job.status      = "done"
        job.finished_at = datetime.now(UTC).isoformat()
        if job.watchdog:
            job.watchdog.stop()
        release_video_id(video_id)
        log.info("[%s] Download worker complete", job.job_id[:8])

    except Exception as exc:
        if job.status != "failed":
            job.error       = str(exc)
            job.status      = "failed"
            job.finished_at = datetime.now(UTC).isoformat()
            if job.watchdog:
                job.watchdog.stop()
            _cleanup_media(video_id)
            release_video_id(video_id)
        log.error("[%s] Download failed: %s", job.job_id[:8], exc)


# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="ytdlp-service",
    description="YouTube metadata, subtitle and audio download microservice",
    version="1.1.0",
)


# ─── Request / Response models ────────────────────────────────────────────────

class DownloadRequest(BaseModel):
    video_id: str
    meta:     bool = False
    human:    bool = False
    auto:     bool = False
    audio:    bool = False


class DownloadResponse(BaseModel):
    job_id:   str
    video_id: str
    status:   str
    message:  str


class JobStatusResponse(BaseModel):
    job_id:          str
    video_id:        str
    status:          str
    title:           Optional[str]
    channel:         Optional[str]
    subtitle_source: Optional[str]
    transcript:      Optional[str]
    audio_path:      Optional[str]
    error:           Optional[str]
    created_at:      str
    finished_at:     Optional[str]


class CancelResponse(BaseModel):
    job_id:  str
    status:  str
    message: str


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "ytdlp-service"}


@app.post("/download", response_model=DownloadResponse)
def start_download(req: DownloadRequest):
    """
    Start a download job for the given video_id.
    Deduplicates: if an active job exists for this video_id, returns it.
    """
    if not VIDEO_ID_RE.fullmatch(req.video_id or ""):
        raise HTTPException(status_code=400, detail="video_id must be an 11-character YouTube ID")

    # ── Deduplication check ───────────────────────────────────────────────────
    existing = get_active_job_for_video(req.video_id)
    if existing and existing.status in ("queued", "running"):
        log.info("[%s] Dedup: returning existing job for video_id=%s",
                 existing.job_id[:8], req.video_id)
        return DownloadResponse(
            job_id=existing.job_id,
            video_id=req.video_id,
            status=existing.status,
            message="Existing download job returned",
        )

    # ── Create new job ────────────────────────────────────────────────────────
    job_id = str(uuid.uuid4())
    job = Job(
        job_id=job_id, video_id=req.video_id,
        meta=req.meta, human=req.human, auto=req.auto, audio=req.audio,
    )
    store_job(job)

    # ── Start per-job watchdog ────────────────────────────────────────────────
    watchdog = JobWatchdog(
        job_id       = job_id,
        poll_timeout = POLL_TIMEOUT_SECONDS,
        job_timeout  = JOB_TIMEOUT_SECONDS,
        on_timeout   = lambda reason: kill_job(job, reason),
    )
    job.watchdog = watchdog
    watchdog.start()

    # ── Start download worker ─────────────────────────────────────────────────
    t = threading.Thread(
        target=download_worker, args=(job,),
        daemon=True, name=f"ytdlp-{job_id[:8]}"
    )
    t.start()

    log.info("[%s] New job created for video_id=%s", job_id[:8], req.video_id)
    return DownloadResponse(
        job_id=job_id,
        video_id=req.video_id,
        status="queued",
        message="Download job started",
    )


@app.get("/download/{job_id}", response_model=JobStatusResponse)
def poll_download(job_id: str):
    """Poll download job status. Resets per-job watchdog timer."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    # Reset watchdog on each poll — client is still alive
    if job.watchdog:
        job.watchdog.reset()

    return JobStatusResponse(
        job_id=job.job_id,
        video_id=job.video_id,
        status=job.status,
        title=job.title,
        channel=job.channel,
        subtitle_source=job.subtitle_source,
        transcript=job.transcript,
        audio_path=job.audio_path,
        error=job.error,
        created_at=job.created_at,
        finished_at=job.finished_at,
    )


@app.delete("/download/{job_id}", response_model=CancelResponse)
def cancel_download(job_id: str):
    """
    Cancel a running download job — kills subprocess and force-removes all
    media files (.mp3, .webm, .part etc). Preserves .txt and .html files.
    """
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    if job.status not in ("queued", "running"):
        # Still force-clean media even on terminal state — client explicitly asked
        _cleanup_media(job.video_id)
        return CancelResponse(
            job_id=job_id,
            status=job.status,
            message=f"Job in terminal state — media files cleaned up",
        )
    # Kill subprocess and mark failed
    proc = job.process
    if proc is not None:
        try:
            if proc.returncode is None:
                proc.kill()
                log.info("[%s] Subprocess killed (client cancel)", job_id[:8])
        except Exception as exc:
            log.warning("[%s] Kill error: %s", job_id[:8], exc)
    job.process     = None
    job.status      = "failed"
    job.error       = "Cancelled by client"
    job.finished_at = datetime.now(UTC).isoformat()
    if job.watchdog:
        job.watchdog.stop()
    release_video_id(job.video_id)
    # Force-remove all media files regardless of completion state
    _cleanup_media(job.video_id)
    return CancelResponse(
        job_id=job_id,
        status="failed",
        message="Download job cancelled — media files cleaned up",
    )
