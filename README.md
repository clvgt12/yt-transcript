# yt-transcribe-web

Streamlit web interface for YouTube transcription + Ollama summarization.

## Quick start

```bash
# Create host directories
mkdir -p ~/yt-transcribe/web/files ~/yt-transcribe/web/whisper

# Configure environment
cp .env.example .env
# Edit .env — add OLLAMA_API_KEY if you have one

# Build and launch
docker compose up -d --build

# Open in browser
open http://localhost:8501
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| WHISPER_MODEL | small | Whisper model: tiny, base, small, medium, large |
| GPU_MODELS | tiny base small | Space-separated models that run on GPU |
| FORCE_WHISPER | false | Skip subtitle check, always use Whisper |
| SUMMARIZE | true | Enable Ollama summarization |
| OLLAMA_MODEL | qwen3:1.7b | Local Ollama model name |
| OLLAMA_CLOUD_MODEL | gpt-oss:20b | Ollama cloud model name |
| OLLAMA_API_KEY | (unset) | Ollama cloud API key |
| FORCE_LOCAL_SUMMARY | false | Skip cloud, use local model only |
| POLL_INTERVAL_MS | 2000 | Browser polling interval (ms) |

## Output files

Written to `~/yt-transcribe/web/files/<video_id>/` on the host:

- `<title>.txt`  — plain text transcript
- `<title>.html` — rendered HTML summary

## Architecture

```
Browser → Streamlit (port 8501) → yt-dlp (subtitles/audio)
                                → Whisper (local GPU transcription)
                                → Ollama cloud API (summarization)
                                → Ollama container (local fallback)
```
