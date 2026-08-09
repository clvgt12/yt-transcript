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

FILES_BASE.mkdir(parents=True, exist_ok=True)

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
    for pattern in ("*.mp3", "*.webm", "*.m4a", "*.part", "*.ytdl",
                    "*.vtt", "*.en.vtt"):
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
    job.process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    stdout, stderr = job.process.communicate()
    rc = job.process.returncode
    job.process = None
    return rc, stdout.strip(), stderr.strip()


def download_worker(job: Job):
    """Execute requested download operations sequentially."""
    job.status = "running"
    video_id   = job.video_id
    out_dir    = FILES_BASE / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("[%s] Starting download worker (meta=%s human=%s auto=%s audio=%s)",
             job.job_id[:8], job.meta, job.human, job.auto, job.audio)

    try:
        # ── 1. Metadata ───────────────────────────────────────────────────────
        if job.meta:
            # Fetch title and channel in one yt-dlp call
            rc, stdout, stderr = _run_cmd(job, [
                YT_DLP_BIN, "--print", "%(title)s	%(channel|uploader|NA)s", "--",
                f"https://www.youtube.com/watch?v={video_id}"
            ])
            if job.status == "failed":
                return
            # Treat JS runtime warning as non-fatal — yt-dlp still extracts
            # metadata for most videos without a JS runtime
            js_warning = "no supported javascript runtime" in stderr.lower()
            if js_warning:
                log.warning("[%s] yt-dlp JS runtime warning (non-fatal): %s",
                            job.job_id[:8], stderr[:120])
            if rc == 0 and stdout:
                parts        = stdout.split("	", 1)
                job.title    = parts[0].strip()
                job.channel  = parts[1].strip() if len(parts) > 1 else None
                if job.channel in ("NA", ""):
                    job.channel = None
                log.info("[%s] Title: %s | Channel: %s",
                         job.job_id[:8], job.title, job.channel)
            elif js_warning and not stdout:
                # JS runtime is blocking extraction entirely for this video
                raise RuntimeError(
                    "YouTube requires a JavaScript runtime for this video. "
                    "Deno is being installed — please retry after container rebuild."
                )
            else:
                raise RuntimeError(f"Metadata fetch failed: {stderr[:200]}")

        # ── 2. Human subtitles ────────────────────────────────────────────────
        if job.human:
            rc, _, _ = _run_cmd(job, [
                YT_DLP_BIN, "--skip-download", "--write-subs",
                "--sub-lang", "en", "--sub-format", "vtt",
                "--output", str(out_dir / "%(title)s.%(ext)s"),
                f"https://www.youtube.com/watch?v={video_id}"
            ])
            if job.status == "failed":
                return
            vtts = [f for f in out_dir.glob("*.en.vtt") if "live_chat" not in f.name]
            if vtts:
                job.transcript      = vtt_to_text(vtts[0])
                job.subtitle_source = "human"
                vtts[0].unlink(missing_ok=True)
                log.info("[%s] Human subtitles downloaded", job.job_id[:8])

        # ── 3. Auto-generated subtitles ───────────────────────────────────────
        if job.auto and not job.transcript:
            rc, _, _ = _run_cmd(job, [
                YT_DLP_BIN, "--skip-download", "--write-auto-subs",
                "--sub-lang", "en", "--sub-format", "vtt",
                "--output", str(out_dir / "%(title)s.%(ext)s"),
                f"https://www.youtube.com/watch?v={video_id}"
            ])
            if job.status == "failed":
                return
            vtts = [f for f in out_dir.glob("*.en.vtt") if "live_chat" not in f.name]
            if vtts:
                job.transcript      = vtt_to_text(vtts[0])
                job.subtitle_source = "auto"
                vtts[0].unlink(missing_ok=True)
                log.info("[%s] Auto subtitles downloaded", job.job_id[:8])
            else:
                job.subtitle_source = "none"

        # ── 4. Audio (mp3) ────────────────────────────────────────────────────
        if job.audio:
            log.info("[%s] Downloading audio...", job.job_id[:8])
            rc, _, stderr = _run_cmd(job, [
                YT_DLP_BIN, "--extract-audio", "--audio-format", "mp3",
                "--audio-quality", "0",
                "--output", str(out_dir / "%(title)s.%(ext)s"),
                f"https://www.youtube.com/watch?v={video_id}"
            ])
            if job.status == "failed":
                return
            if rc != 0:
                raise RuntimeError(f"Audio download failed: {stderr[:200]}")
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
    version="1.0.0",
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
    if not req.video_id:
        raise HTTPException(status_code=400, detail="video_id is required")

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
