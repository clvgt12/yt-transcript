"""
whisper-service — FastAPI microservice for audio transcription via OpenAI Whisper.

API:
    POST /transcribe
        Accepts a multipart audio file upload.
        Returns a job_id immediately. Starts a per-job watchdog timer.

    GET /transcribe/{job_id}
        Poll for job status and result. Resets the watchdog timer.

    DELETE /transcribe/{job_id}
        Cancel a running job immediately.

    GET /health
        Health check endpoint.

Per-job watchdog: started on POST, reset on each GET poll, fires if no poll
received within POLL_TIMEOUT_SECONDS — kills the whisper subprocess cleanly.
"""

import os
import sys
import uuid
import logging
import threading
import subprocess
import tempfile
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("whisper-service")

# ─── Configuration ─────────────────────────────────────────────────────────────

WHISPER_MODEL        = os.environ.get("WHISPER_MODEL",       "small")
GPU_MODELS           = os.environ.get("GPU_MODELS",          "tiny base small").split()
VENV_PATH            = os.environ.get("VENV_PATH",           "/venv")
WHISPER_CACHE        = os.environ.get("WHISPER_CACHE",       "/whisper-cache")
JOB_TIMEOUT_SECONDS  = int(os.environ.get("JOB_TIMEOUT_SECONDS", "600"))

# Derive poll timeout from polling interval: 3 missed cycles + 1s margin
_POLL_INTERVAL_MS    = int(os.environ.get("POLL_INTERVAL_MS", "2000"))
POLL_TIMEOUT_SECONDS = (3 * _POLL_INTERVAL_MS // 1000) + 1

Path(WHISPER_CACHE).mkdir(parents=True, exist_ok=True)

# ─── Per-job watchdog ─────────────────────────────────────────────────────────

class JobWatchdog:
    """
    Per-job countdown timer running in its own thread.
    Started when a transcription job is created (POST /transcribe).
    Reset to initial value on each client poll (GET /transcribe/{job_id}).
    Fires and kills the whisper subprocess if the countdown reaches zero —
    indicating the client has disconnected or stopped polling.
    Also enforces a hard maximum job duration (JOB_TIMEOUT_SECONDS).
    """
    def __init__(self, job_id: str, poll_timeout: int, job_timeout: int,
                 on_timeout):
        self.job_id       = job_id
        self.poll_timeout = poll_timeout   # seconds before firing on poll silence
        self.job_timeout  = job_timeout    # hard cap regardless of polls
        self.on_timeout   = on_timeout     # callback: fn(reason: str)
        self._remaining   = poll_timeout   # countdown value in seconds
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
        """Reset countdown — call on each client poll."""
        with self._lock:
            self._remaining = self.poll_timeout

    def stop(self):
        """Stop the watchdog — call when job completes or is cancelled."""
        self._stopped.set()

    def _run(self):
        job_start   = time.time()
        tick        = 1   # check every second

        while not self._stopped.is_set():
            time.sleep(tick)

            # Hard job timeout
            if time.time() - job_start >= self.job_timeout:
                log.warning("[%s] Watchdog: hard job timeout (%ds)",
                            self.job_id[:8], self.job_timeout)
                self.on_timeout("job timeout")
                return

            # Poll silence countdown
            with self._lock:
                self._remaining -= tick
                remaining = self._remaining

            if remaining <= 0:
                log.warning("[%s] Watchdog: client disconnected (no poll for %ds)",
                            self.job_id[:8], self.poll_timeout)
                self.on_timeout("client disconnected")
                return

        log.info("[%s] Watchdog stopped", self.job_id[:8])


# ─── Job store ────────────────────────────────────────────────────────────────

class Job:
    def __init__(self, job_id: str, filename: str, model: str):
        self.job_id      = job_id
        self.filename    = filename
        self.model       = model
        self.status      = "queued"   # queued | running | done | failed
        self.transcript: Optional[str] = None
        self.error: Optional[str]      = None
        self.created_at  = datetime.now(UTC).isoformat()
        self.finished_at: Optional[str] = None
        self.process: Optional[subprocess.Popen] = None
        self.watchdog: Optional[JobWatchdog]     = None


_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()


def get_job(job_id: str) -> Optional[Job]:
    with _jobs_lock:
        return _jobs.get(job_id)


def store_job(job: Job):
    with _jobs_lock:
        _jobs[job.job_id] = job


# ─── Kill helper ──────────────────────────────────────────────────────────────

def kill_job(job: Job, reason: str):
    """Kill the whisper subprocess and mark job as failed."""
    proc = job.process
    if proc is not None:
        try:
            if proc.returncode is None:
                proc.kill()
                log.info("[%s] Subprocess killed (%s)", job.job_id[:8], reason)
        except Exception as exc:
            log.warning("[%s] Kill failed: %s", job.job_id[:8], exc)
    job.process     = None
    job.status      = "failed"
    job.error       = f"Job terminated: {reason}"
    job.finished_at = datetime.now(UTC).isoformat()
    if job.watchdog:
        job.watchdog.stop()


# ─── Transcription worker ─────────────────────────────────────────────────────

def transcribe_worker(job: Job, audio_path: Path):
    """Run Whisper transcription in a background thread."""
    job.status = "running"
    log.info("[%s] Starting transcription — model=%s file=%s",
             job.job_id[:8], job.model, audio_path.name)

    try:
        device   = "cuda" if job.model in GPU_MODELS else "cpu"
        out_dir  = audio_path.parent
        activate = Path(VENV_PATH) / "bin" / "activate"

        cmd = (
            f"source {activate} && "
            f"XDG_CACHE_HOME={WHISPER_CACHE} "
            f"whisper '{audio_path}' "
            f"--model {job.model} "
            f"--device {device} "
            f"--output_dir '{out_dir}' "
            f"--output_format txt "
            f"--verbose False"
        )

        job.process = subprocess.Popen(
            cmd, shell=True, executable="/bin/bash",
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        stdout, stderr = job.process.communicate()
        result_rc  = job.process.returncode
        job.process = None

        # Check if killed by watchdog while communicate() was blocking
        if job.status == "failed":
            log.info("[%s] Job was cancelled during transcription", job.job_id[:8])
            return

        if result_rc != 0:
            raise RuntimeError(stderr.strip() or "Whisper exited non-zero")

        txts = list(out_dir.glob("*.txt"))
        if not txts:
            raise RuntimeError("Whisper produced no .txt output")

        job.transcript  = txts[0].read_text(encoding="utf-8")
        job.status      = "done"
        job.finished_at = datetime.now(UTC).isoformat()
        if job.watchdog:
            job.watchdog.stop()
        log.info("[%s] Transcription complete — %d chars",
                 job.job_id[:8], len(job.transcript))

    except Exception as exc:
        if job.status != "failed":   # don't overwrite watchdog-set status
            job.error       = str(exc)
            job.status      = "failed"
            job.finished_at = datetime.now(UTC).isoformat()
            if job.watchdog:
                job.watchdog.stop()
        log.error("[%s] Transcription failed: %s", job.job_id[:8], exc)

    finally:
        job.process = None
        try:
            for f in audio_path.parent.iterdir():
                f.unlink(missing_ok=True)
            audio_path.parent.rmdir()
        except Exception:
            pass


# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="whisper-service",
    description="Async audio transcription via OpenAI Whisper",
    version="1.0.0",
)


class TranscribeResponse(BaseModel):
    job_id:  str
    status:  str
    message: str


class CancelResponse(BaseModel):
    job_id:  str
    status:  str
    message: str


class JobStatusResponse(BaseModel):
    job_id:      str
    status:      str
    model:       str
    filename:    str
    transcript:  Optional[str]
    error:       Optional[str]
    created_at:  str
    finished_at: Optional[str]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model":  WHISPER_MODEL,
        "device": "cuda" if WHISPER_MODEL in GPU_MODELS else "cpu",
    }


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    file:  UploadFile = File(...),
    model: str = None,
):
    """
    Accept an audio file upload and start async transcription.
    Starts a per-job watchdog timer. Returns job_id for polling.
    """
    use_model = model or WHISPER_MODEL

    if use_model not in ("tiny", "base", "small", "medium", "large"):
        raise HTTPException(status_code=400, detail=f"Invalid model: {use_model}")

    job_id     = str(uuid.uuid4())
    tmp_dir    = Path(tempfile.mkdtemp(prefix=f"whisper_{job_id[:8]}_"))
    suffix     = Path(file.filename).suffix or ".mp3"
    audio_path = tmp_dir / f"audio{suffix}"

    content = await file.read()
    audio_path.write_bytes(content)

    log.info("[%s] Received %s (%d bytes) model=%s",
             job_id[:8], file.filename, len(content), use_model)

    job = Job(job_id=job_id, filename=file.filename, model=use_model)
    store_job(job)

    # Start per-job watchdog
    watchdog = JobWatchdog(
        job_id       = job_id,
        poll_timeout = POLL_TIMEOUT_SECONDS,
        job_timeout  = JOB_TIMEOUT_SECONDS,
        on_timeout   = lambda reason: kill_job(job, reason),
    )
    job.watchdog = watchdog
    watchdog.start()

    # Start transcription worker
    thread = threading.Thread(
        target=transcribe_worker,
        args=(job, audio_path),
        daemon=True,
        name=f"whisper-{job_id[:8]}",
    )
    thread.start()

    return TranscribeResponse(
        job_id=job_id,
        status="queued",
        message=f"Transcription started with model '{use_model}'",
    )


@app.delete("/transcribe/{job_id}", response_model=CancelResponse)
def cancel_transcription(job_id: str):
    """Cancel a queued or running transcription job immediately."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    if job.status not in ("queued", "running"):
        return CancelResponse(
            job_id=job_id,
            status=job.status,
            message=f"Job already in terminal state: {job.status}",
        )
    kill_job(job, "cancelled by client")
    return CancelResponse(
        job_id=job_id,
        status="failed",
        message="Job cancelled",
    )


@app.get("/transcribe/{job_id}", response_model=JobStatusResponse)
def get_transcription(job_id: str):
    """Poll transcription job status. Resets the per-job watchdog timer."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    # Reset watchdog countdown on each poll — client is still alive
    if job.watchdog:
        job.watchdog.reset()

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        model=job.model,
        filename=job.filename,
        transcript=job.transcript,
        error=job.error,
        created_at=job.created_at,
        finished_at=job.finished_at,
    )
