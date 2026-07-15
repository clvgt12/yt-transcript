"""
whisper-service — FastAPI microservice for audio transcription via OpenAI Whisper.

API:
    POST /transcribe
        Accepts a multipart audio file upload.
        Returns a job_id immediately.

    GET /transcribe/{job_id}
        Poll for job status and result.

    GET /health
        Health check endpoint.

Jobs are stored in-process (dict). The service is single-user / home-lab
scoped — no persistence across restarts by design.
"""

import os
import sys
import uuid
import logging
import threading
import subprocess
import tempfile
from datetime import datetime, UTC
from pathlib import Path
from typing import Optional

import time
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

WHISPER_MODEL    = os.environ.get("WHISPER_MODEL",    "small")
GPU_MODELS       = os.environ.get("GPU_MODELS",       "tiny base small").split()
VENV_PATH        = os.environ.get("VENV_PATH",        "/venv")
WHISPER_CACHE        = os.environ.get("WHISPER_CACHE",        "/whisper-cache")
JOB_TIMEOUT_SECONDS  = int(os.environ.get("JOB_TIMEOUT_SECONDS",  "600"))  # 10 min
# Derive poll timeout from the web app's polling interval:
# assume client disconnected if no poll received within 3 polling cycles + 1s margin
_POLL_INTERVAL_MS    = int(os.environ.get("POLL_INTERVAL_MS",     "2000"))
POLL_TIMEOUT_SECONDS = (3 * _POLL_INTERVAL_MS // 1000) + 1

Path(WHISPER_CACHE).mkdir(parents=True, exist_ok=True)

# ─── Job store ────────────────────────────────────────────────────────────────

class Job:
    def __init__(self, job_id: str, filename: str, model: str):
        self.job_id     = job_id
        self.filename   = filename
        self.model      = model
        self.status     = "queued"   # queued | running | done | failed
        self.transcript: Optional[str] = None
        self.error: Optional[str]      = None
        self.created_at  = datetime.now(UTC).isoformat()
        self.created_ts  = time.time()   # float epoch for watchdog
        self.finished_at: Optional[str] = None
        self.last_polled = time.time()   # updated on each GET poll
        self.process: Optional[subprocess.Popen] = None  # subprocess handle

_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()

def get_job(job_id: str) -> Optional[Job]:
    with _jobs_lock:
        return _jobs.get(job_id)

def store_job(job: Job):
    with _jobs_lock:
        _jobs[job.job_id] = job

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

        if result_rc != 0:
            raise RuntimeError(stderr.strip() or "Whisper exited non-zero")

        # Find the generated .txt file
        txts = list(out_dir.glob("*.txt"))
        if not txts:
            raise RuntimeError("Whisper produced no .txt output")

        job.transcript  = txts[0].read_text(encoding="utf-8")
        job.status      = "done"
        job.finished_at = datetime.now(UTC).isoformat()
        log.info("[%s] Transcription complete — %d chars",
                 job.job_id[:8], len(job.transcript))

    except Exception as exc:
        job.error       = str(exc)
        job.status      = "failed"
        job.finished_at = datetime.now(UTC).isoformat()
        log.error("[%s] Transcription failed: %s", job.job_id[:8], exc)

    finally:
        job.process = None
        # Clean up temp audio file and any whisper output files
        try:
            for f in audio_path.parent.iterdir():
                f.unlink(missing_ok=True)
            audio_path.parent.rmdir()
        except Exception:
            pass

# ─── FastAPI app ──────────────────────────────────────────────────────────────

def watchdog():
    """
    Background thread — checks running jobs every 30 seconds.
    Kills a job if:
      - Running longer than JOB_TIMEOUT_SECONDS (hard cap), OR
      - No poll received in POLL_TIMEOUT_SECONDS (client disconnected /
        Streamlit Stop button pressed)
    """
    # Check interval = 1 polling cycle (derived directly from POLL_INTERVAL_MS)
    check_interval = max(5, _POLL_INTERVAL_MS // 1000)
    log.info("Watchdog started (job_timeout=%ds, poll_timeout=%ds, check_interval=%ds)",
             JOB_TIMEOUT_SECONDS, POLL_TIMEOUT_SECONDS, check_interval)
    while True:
        time.sleep(check_interval)
        now = time.time()
        with _jobs_lock:
            jobs = list(_jobs.values())
        for job in jobs:
            if job.status != "running":
                continue
            age     = now - job.created_ts
            no_poll = now - job.last_polled

            timed_out = age     > JOB_TIMEOUT_SECONDS
            abandoned = no_poll > POLL_TIMEOUT_SECONDS

            if timed_out or abandoned:
                reason = "timeout" if timed_out else "client disconnected"
                log.warning("Watchdog killing job %s (%s)", job.job_id[:8], reason)
                if job.process and job.process.poll() is None:
                    job.process.kill()
                    log.info("Subprocess killed for job %s", job.job_id[:8])
                job.status      = "failed"
                job.error       = f"Job terminated by watchdog ({reason})"
                job.finished_at = datetime.now(UTC).isoformat()
                job.process     = None


app = FastAPI(
    title="whisper-service",
    description="Async audio transcription via OpenAI Whisper",
    version="1.0.0",
)


@app.on_event("startup")
def startup_event():
    t = threading.Thread(target=watchdog, daemon=True, name="watchdog")
    t.start()
    log.info("whisper-service ready — watchdog running")


class TranscribeResponse(BaseModel):
    job_id: str
    status: str
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
    return {"status": "ok", "model": WHISPER_MODEL, "device": "cuda" if WHISPER_MODEL in GPU_MODELS else "cpu"}


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    file:  UploadFile = File(...),
    model: str = None,
):
    """
    Accept an audio file upload and start async transcription.
    Returns job_id for polling via GET /transcribe/{job_id}.
    """
    use_model = model or WHISPER_MODEL

    if use_model not in ("tiny", "base", "small", "medium", "large"):
        raise HTTPException(status_code=400, detail=f"Invalid model: {use_model}")

    # Save uploaded file to a temp directory
    job_id   = str(uuid.uuid4())
    tmp_dir  = Path(tempfile.mkdtemp(prefix=f"whisper_{job_id[:8]}_"))
    suffix   = Path(file.filename).suffix or ".mp3"
    audio_path = tmp_dir / f"audio{suffix}"

    content = await file.read()
    audio_path.write_bytes(content)

    log.info("[%s] Received %s (%d bytes) model=%s",
             job_id[:8], file.filename, len(content), use_model)

    job = Job(job_id=job_id, filename=file.filename, model=use_model)
    store_job(job)

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


@app.get("/transcribe/{job_id}", response_model=JobStatusResponse)
def get_transcription(job_id: str):
    """Poll transcription job status and retrieve result when done."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    # Update last_polled for watchdog client-disconnect detection
    job.last_polled = time.time()

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
