# yt-transcript

Turn any YouTube video into a clean transcript and an AI-generated summary — paste a URL, get readable output in under a minute. Runs as a self-contained Docker stack with GPU-accelerated local transcription, so it works even on videos with no captions and without sending audio to a third party.

## Why this exists

Most "summarize this YouTube video" tools either require captions to already exist or ship your audio off to a SaaS API. This project doesn't assume either:

- **Prefers YouTube's own captions** (human-authored first, auto-generated second) when they're available — fast, free, no local compute needed.
- **Falls back to local Whisper transcription** of the downloaded audio when no captions exist, or whenever you explicitly want local transcription regardless of captions (`FORCE_WHISPER=true`).
- **Summarizes via Ollama** — a cloud model for speed and quality (e.g. `gpt-oss:120b-cloud`), or a fully local fallback model, your choice.
- **GPU-accelerated on either NVIDIA or Intel hardware** — the stack auto-detects which GPU backend your host actually has and builds accordingly. No manual driver-stack guessing.

## Architecture

Four containers, one internal Docker network, one exposed port:

| Service | Role | Exposed? |
|---|---|---|
| `web` | Streamlit UI — orchestrates the workflow, renders results | `:8501` (host) |
| `ytdlp` | Downloads video metadata, captions, or audio via yt-dlp | internal only |
| `whisper` | GPU-accelerated audio transcription | internal only |
| `ollama` | LLM inference for summarization (cloud-proxied or local) | internal only |

The `whisper` service ships as **two interchangeable Docker images** built from the same application code — one using NVIDIA CUDA (`Dockerfile.cuda`), one using Intel OpenVINO (`Dockerfile.openvino`) — selected automatically based on which GPU the build host actually has.

## Prerequisites

- **Docker Engine + the Compose v2 plugin** (`docker compose version` should report v2.x)
- **An Ollama Cloud account and API key** — needed for cloud-model summarization. Free to create at [ollama.com/settings/keys](https://ollama.com/settings/keys). (You can skip this and run fully local-only — see `OLLAMA_NO_CLOUD` below — but a key is required for the default configuration.)
- **A supported GPU, one of:**
  - **NVIDIA** — a CUDA-capable GPU with the proprietary driver and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed on the host
  - **Intel integrated graphics** — a CPU with Intel Iris Xe/UHD graphics (or newer), with the host's `render` group correctly mapping to `/dev/dri/renderD128` (see step 3 below)
- A Linux host — developed and tested on Ubuntu/Ubuntu Studio 24.04 and 26.04 LTS

## Installation

### 1. Clone the repository

```bash
git clone git@github.com:clvgt12/yt-transcript.git
cd yt-transcript
```

### 2. Configure environment variables

```bash
cp env.example .env
```

Edit `.env` and set, at minimum:

```
OLLAMA_API_KEY=your-key-from-ollama.com
```

See [Configuration](#configuration) below for every available option.

### 3. Intel GPU hosts only — set your render group GID

The Intel build needs the host's numeric `render` group ID to grant the container GPU device access. This value is host-specific — **do not skip this on Intel hardware, and don't reuse a GID from another machine.**

```bash
getent group render
```

Add the number after the second colon to `.env`:

```
RENDER_GID=990    # whatever your host actually reports
```

NVIDIA hosts can skip this step entirely.

### 4. Build the containers

The included management script detects your GPU automatically (checks for a working `nvidia-smi` first, then falls back to checking for an Intel render node) and builds the matching image:

```bash
./yt-transcribe.sh build
```

First build takes several minutes — it's compiling/downloading the full CUDA or OpenVINO toolchain. Use `--no-cache` if you need a clean rebuild after changing a Dockerfile:

```bash
./yt-transcribe.sh build --no-cache
```

If you'd rather bypass auto-detection or the script entirely:

```bash
YT_TRANSCRIBE_GPU=intel ./yt-transcribe.sh build   # force a backend
# or, raw docker compose:
docker compose -f docker-compose.yml -f docker-compose.cuda.yml build     # NVIDIA
docker compose -f docker-compose.yml -f docker-compose.intel.yml build    # Intel
```

### 5. Start the stack

```bash
./yt-transcribe.sh start
```

### 6. Open the app

```
http://localhost:8501
```

Paste a YouTube URL and go.

## Usage — `yt-transcribe.sh`

| Command | What it does |
|---|---|
| `start` | Detect GPU backend, bring the stack up (`up -d`) |
| `stop` | Bring the stack down |
| `restart` | `stop`, then `start` |
| `build [args...]` | Build images; extra args pass through (e.g. `--no-cache`) |
| `clean` | `docker system prune -f` — **host-wide**, not scoped to this project |
| `realclean` | `stop`, then `clean` |

Override GPU auto-detection anywhere with `YT_TRANSCRIBE_GPU=cuda|intel`.

## Configuration

All settings live in `.env` (see `env.example` for the full annotated template).

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_API_KEY` | *(required)* | Cloud model access — [get one here](https://ollama.com/settings/keys) |
| `OLLAMA_PRIMARY_MODEL` | `gpt-oss:120b-cloud` | Summarization model, tried first |
| `OLLAMA_FALLBACK_MODEL` | `qwen3:1.7b` | Local model, used if the primary fails or cloud is disabled |
| `OLLAMA_NO_CLOUD` | `0` | Set `1` to force fully local summarization, no cloud calls at all |
| `FORCE_LOCAL_SUMMARY` | *(unset)* | Same effect as above, alternate switch |
| `OLLAMA_CONTEXT_LENGTH` | `24576` | Context window (tokens) for the local Ollama model |
| `FORCE_WHISPER` | *(unset)* | Force local Whisper transcription even when YouTube captions exist |
| `WHISPER_MODEL` | `small` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large` |
| `GPU_MODELS` | `tiny base small` | Which model sizes run on GPU; anything else falls back to CPU (VRAM/memory limits) |
| `TARGET_LANG` | `auto` | `auto` transcribes in the detected source language; `en` translates any source language to English. Whisper cannot translate to any other target language — that's a model limitation, not a config option. |
| `WEB_SEARCH_ENABLED` | `true` | Enables agentic web search for follow-up questions about a summarized video |
| `WEB_SEARCH_MAX_RESULTS` | `3` | Search results per follow-up query |
| `JOB_TIMEOUT_SECONDS` | `600` | Hard timeout for a single transcription job |
| `POLL_INTERVAL_MS` | `2000` | UI/service polling interval |
| `CACHE_FILE_AGE_DAYS` | `7` | Auto-expire cached downloads/output after N days (`0` disables) |
| `RENDER_GID` | *(Intel only)* | Host's `render` group GID — see installation step 3 |

## Data and caching

- Transcripts and summaries are written to `~/yt-transcribe/web/files/<video_id>/` on the host.
- Whisper model weights and, on the Intel backend, converted OpenVINO IR models are cached under `~/.cache/` so rebuilds and restarts don't re-download or re-convert from scratch.
- The Ollama model store is a named Docker volume (`ollama`), reused across rebuilds.

## Troubleshooting

- **First OpenVINO build is slow, first transcription job is *very* slow** — the first request against a given Whisper model size triggers a one-time conversion to OpenVINO's IR format, which is cached afterward. Subsequent jobs are fast.
- **Container builds but GPU isn't actually used** — check `docker logs yt-transcribe-whisper` for the startup line reporting `backend=` and `device=`, and confirm via `GET /health` on the whisper service. On NVIDIA, watch `nvidia-smi`/`nvtop` during a job; on Intel, `intel_gpu_top`.
- **Intel builds fail on `docker exec ... groups`** showing an unresolved GID — cosmetic only; `group_add` grants by numeric GID at the kernel level and doesn't require a name in the container's `/etc/group`. Verify with `docker exec yt-transcribe-whisper id` instead.

## License

*(Add a `LICENSE` file to the repository and reference it here.)*
