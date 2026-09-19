"""
whisper-service — FastAPI microservice for audio transcription via Whisper.

Supports two interchangeable backends, selected via WHISPER_BACKEND:
    cuda      — shells out to the `whisper` CLI (openai-whisper package),
                --device cuda|cpu. Unchanged from the original kamakazi
                implementation.
    openvino  — in-process inference via optimum-intel's
                OVModelForSpeechSeq2Seq, targeting OPENVINO_DEVICE
                (GPU|CPU). Used on tepache's Intel iGPU.

WHISPER_BACKEND defaults to "openvino" if OPENVINO_DEVICE is set, else
"cuda" — but each Dockerfile should set it explicitly (ENV WHISPER_BACKEND=
cuda / openvino) rather than relying on the fallback.

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

Cancellation note (openvino backend only): a running OVModelForSpeechSeq2Seq
.generate() call cannot be interrupted mid-inference the way a subprocess can
be killed. Cancelling an openvino job marks it failed immediately for the
client, but the underlying inference call keeps running in its background
thread until it finishes — the result is simply discarded (see the
job.status == "failed" check in transcribe_worker). This differs from the
cuda backend, where cancellation kills the subprocess outright.
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

# Whisper supports exactly two modes: "auto" (transcribe in the detected
# source language) or "en" (translate — Whisper only ever translates TO
# English, never to an arbitrary target language; that's a hard model
# limitation, not a config option). Any other value would silently force
# transcription to assume the wrong source language rather than translate
# anything, so it's rejected here rather than passed through.
TARGET_LANG = os.environ.get("TARGET_LANG", "auto").strip().lower()
if TARGET_LANG not in ("auto", "en"):
    log.warning("TARGET_LANG=%r is not supported (Whisper can only "
                "auto-detect/transcribe or translate to English) — "
                "falling back to 'auto'", TARGET_LANG)
    TARGET_LANG = "auto"

# OpenVINO-specific (ignored by the cuda backend)
OPENVINO_DEVICE      = os.environ.get("OPENVINO_DEVICE",     "CPU")

# Backend selection — explicit env var wins; falls back to inferring from
# OPENVINO_DEVICE only if WHISPER_BACKEND wasn't set. Set it explicitly in
# each Dockerfile (ENV WHISPER_BACKEND=cuda / openvino) rather than relying
# on this fallback.
WHISPER_BACKEND = os.environ.get("WHISPER_BACKEND") or (
    "openvino" if "OPENVINO_DEVICE" in os.environ else "cuda"
)
if WHISPER_BACKEND not in ("cuda", "openvino"):
    raise RuntimeError(
        f"Unsupported WHISPER_BACKEND: {WHISPER_BACKEND!r} "
        f"(expected 'cuda' or 'openvino')"
    )

# Derive poll timeout from polling interval: 3 missed cycles + 1s margin
_POLL_INTERVAL_MS    = int(os.environ.get("POLL_INTERVAL_MS", "2000"))
POLL_TIMEOUT_SECONDS = (3 * _POLL_INTERVAL_MS // 1000) + 1

Path(WHISPER_CACHE).mkdir(parents=True, exist_ok=True)

log.info("whisper-service starting — backend=%s device=%s model=%s",
         WHISPER_BACKEND,
         OPENVINO_DEVICE if WHISPER_BACKEND == "openvino" else "cuda/cpu (per-model)",
         WHISPER_MODEL)

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
        self.process: Optional[subprocess.Popen] = None   # cuda backend only
        self.watchdog: Optional[JobWatchdog]     = None


_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()


def get_job(job_id: str) -> Optional[Job]:
    with _jobs_lock:
        return _jobs.get(job_id)


def store_job(job: Job):
    with _jobs_lock:
        _jobs[job.job_id] = job


# ─── Device resolution ─────────────────────────────────────────────────────────

def resolve_device(model_name: str) -> str:
    """
    Which compute device a given model size should run on, for the active
    backend. Mirrors the original GPU_MODELS logic: models in GPU_MODELS get
    the accelerator; everything else (typically medium/large, which don't
    fit in 4GB VRAM or a laptop iGPU's shared memory pool) falls back to CPU.
    """
    if WHISPER_BACKEND == "cuda":
        return "cuda" if model_name in GPU_MODELS else "cpu"
    else:  # openvino
        return OPENVINO_DEVICE if model_name in GPU_MODELS else "CPU"


# ─── Kill helper ──────────────────────────────────────────────────────────────

def kill_job(job: Job, reason: str):
    """
    Mark a job failed and, for the cuda backend, kill its subprocess.
    For the openvino backend there is no subprocess to kill — the in-process
    generate() call is left to finish in its background thread and its
    result is discarded once it notices job.status == "failed"
    (see _transcribe_openvino's caller in transcribe_worker).
    """
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


# ─── CUDA backend — subprocess via the `whisper` CLI ──────────────────────────

def _transcribe_cuda(job: Job, audio_path: Path):
    """
    Original kamakazi implementation, unchanged: shells out to the
    openai-whisper CLI. Sets job.transcript/status directly (rather than
    returning a value) because it needs to track job.process for
    cancellation via kill_job().
    """
    device   = resolve_device(job.model)
    out_dir  = audio_path.parent
    activate = Path(VENV_PATH) / "bin" / "activate"

    # TARGET_LANG="en" -> Whisper's translate task (any source language ->
    # English). "auto" -> default transcribe task, no flag needed; Whisper
    # already auto-detects the source language on its own.
    task_flag = "--task translate " if TARGET_LANG == "en" else ""

    cmd = (
        f"source {activate} && "
        f"XDG_CACHE_HOME={WHISPER_CACHE} "
        f"whisper '{audio_path}' "
        f"--model {job.model} "
        f"--device {device} "
        f"{task_flag}"
        f"--output_dir '{out_dir}' "
        f"--output_format txt "
        f"--verbose False"
    )

    job.process = subprocess.Popen(
        cmd, shell=True, executable="/bin/bash",
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    stdout, stderr = job.process.communicate()
    result_rc   = job.process.returncode
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


# ─── OpenVINO backend — in-process via optimum-intel ──────────────────────────

# HF model repo IDs for each supported size. "large" maps to large-v3 (the
# current recommended checkpoint) rather than the older large/large-v2.
_HF_MODEL_IDS = {
    "tiny":   "openai/whisper-tiny",
    "base":   "openai/whisper-base",
    "small":  "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large":  "openai/whisper-large-v3",
}

_ov_models_lock = threading.Lock()
_ov_models: dict[str, tuple] = {}   # "{model}:{device}" -> (ov_model, processor)

# OpenVINO's GPU plugin is not safe for concurrent inference calls sharing
# one device context — two simultaneous generate() calls have been observed
# to hang the entire process (not just the two jobs involved), consistent
# with the native blocking wait not releasing the GIL. There's only one
# physical GPU regardless of which model size is in use, so this lock is
# global across all models/devices, not per cache_key.
_ov_inference_lock = threading.Lock()


def _load_openvino_model(model_name: str, device: str):
    """
    Load (or fetch from the in-process cache) an OpenVINO-converted Whisper
    model + processor. First load for a given model/device pair converts
    from the HF checkpoint to OpenVINO IR and saves it under
    WHISPER_CACHE/openvino-ir/ (which lives on the same bind-mounted volume
    as the CUDA backend's .pt weights) so restarts reuse the converted model
    instead of re-exporting — export is slow, several minutes for 'small'
    and up on this hardware class.
    """
    # Imported here, not at module level, so the cuda image (which never
    # installs optimum-intel/librosa) doesn't fail on import.
    from optimum.intel.openvino import OVModelForSpeechSeq2Seq
    from transformers import AutoProcessor

    cache_key = f"{model_name}:{device}"
    with _ov_models_lock:
        if cache_key in _ov_models:
            return _ov_models[cache_key]

        hf_id = _HF_MODEL_IDS.get(model_name)
        if hf_id is None:
            raise ValueError(f"No OpenVINO model mapping for '{model_name}'")

        ir_dir = Path(WHISPER_CACHE) / "openvino-ir" / f"{model_name}-{device.lower()}"

        if ir_dir.exists():
            log.info("[openvino] Loading cached IR for '%s' (%s) from %s",
                      model_name, device, ir_dir)
            model     = OVModelForSpeechSeq2Seq.from_pretrained(ir_dir, device=device)
            processor = AutoProcessor.from_pretrained(ir_dir)
        else:
            log.info("[openvino] Converting '%s' (%s) to OpenVINO IR on device=%s "
                      "— first run, this is slow", model_name, hf_id, device)
            model     = OVModelForSpeechSeq2Seq.from_pretrained(
                hf_id, export=True, device=device
            )
            processor = AutoProcessor.from_pretrained(hf_id)
            ir_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ir_dir)
            processor.save_pretrained(ir_dir)
            log.info("[openvino] Cached IR for '%s' (%s) at %s",
                      model_name, device, ir_dir)

        _ov_models[cache_key] = (model, processor)
        return model, processor


def _transcribe_openvino(job: Job, audio_path: Path):
    """
    In-process transcription via optimum-intel. Sets job.transcript/status
    directly, mirroring _transcribe_cuda's contract, so transcribe_worker
    can treat both backends identically.
    """
    import librosa

    device = resolve_device(job.model)
    model, processor = _load_openvino_model(job.model, device)

    audio, _ = librosa.load(str(audio_path), sr=16000, mono=True)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

    # Serialize the actual GPU call — see _ov_inference_lock's comment.
    # Audio decode/preprocessing above happens outside the lock so it can
    # overlap with another job's inference.
    if not _ov_inference_lock.acquire(blocking=False):
        log.info("[%s] Waiting for GPU (another transcription in progress)...",
                 job.job_id[:8])
        _ov_inference_lock.acquire()
    try:
        if job.status == "failed":
            log.info("[%s] Job was cancelled while waiting for GPU", job.job_id[:8])
            return
        # TARGET_LANG="en" -> translate (any source language -> English).
        # "auto" -> default transcribe, source language auto-detected.
        gen_kwargs = {"task": "translate"} if TARGET_LANG == "en" else {}
        predicted_ids = model.generate(inputs["input_features"], **gen_kwargs)
    finally:
        _ov_inference_lock.release()

    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

    # Check if cancelled while generate() was blocking (see kill_job's
    # docstring — the call itself can't be interrupted, so this is the
    # earliest point we can notice and discard a stale result).
    if job.status == "failed":
        log.info("[%s] Job was cancelled during transcription", job.job_id[:8])
        return

    if not text:
        raise RuntimeError("OpenVINO Whisper produced no output")

    job.transcript  = text
    job.status      = "done"
    job.finished_at = datetime.now(UTC).isoformat()
    if job.watchdog:
        job.watchdog.stop()
    log.info("[%s] Transcription complete — %d chars",
             job.job_id[:8], len(job.transcript))


# ─── Transcription worker ─────────────────────────────────────────────────────

def transcribe_worker(job: Job, audio_path: Path):
    """Run Whisper transcription in a background thread, dispatched by backend."""
    job.status = "running"
    log.info("[%s] Starting transcription — backend=%s model=%s file=%s",
             job.job_id[:8], WHISPER_BACKEND, job.model, audio_path.name)

    try:
        if WHISPER_BACKEND == "cuda":
            _transcribe_cuda(job, audio_path)
        else:
            _transcribe_openvino(job, audio_path)

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
    description="Async audio transcription via Whisper (cuda or openvino backend)",
    version="1.1.0",
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
        "status":      "ok",
        "backend":     WHISPER_BACKEND,
        "model":       WHISPER_MODEL,
        "device":      resolve_device(WHISPER_MODEL),
        "target_lang": TARGET_LANG,
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
