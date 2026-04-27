# yt-transcribe-web Dockerfile
# Base: Ubuntu 24.04 LTS — matches kamakazi host OS
FROM ubuntu:24.04

LABEL maintainer="chris@kamakazi"
LABEL description="YouTube transcription + summarization web UI (Streamlit)"

ARG DEBIAN_FRONTEND=noninteractive

# ─── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    curl \
    jq \
    wget \
    perl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ─── Install yt-dlp (latest binary — avoids stale apt package) ───────────────
RUN wget -q "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp" \
    -O /usr/local/bin/yt-dlp \
    && chmod 755 /usr/local/bin/yt-dlp

# ─── Python virtual environment ───────────────────────────────────────────────
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# ─── PyTorch pinned to 2.2.0+cu118 (last build supporting Pascal/sm_61) ────────
RUN pip install --no-cache-dir     torch==2.2.0+cu118     torchvision==0.17.0+cu118     torchaudio==2.2.0+cu118     --index-url https://download.pytorch.org/whl/cu118

# ─── Remaining Python dependencies from PyPI ─────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ─── App files ────────────────────────────────────────────────────────────────
WORKDIR /usr/app
COPY app.py /usr/app/app.py

# ─── Output directory ─────────────────────────────────────────────────────────
RUN mkdir -p /usr/app/files
VOLUME ["/usr/app/files"]

# ─── Whisper model cache ──────────────────────────────────────────────────────
RUN mkdir -p /root/.cache/whisper
VOLUME ["/root/.cache/whisper"]

# ─── Streamlit config ─────────────────────────────────────────────────────────
RUN mkdir -p /root/.streamlit
COPY streamlit_config.toml /root/.streamlit/config.toml

# ─── Environment defaults ─────────────────────────────────────────────────────
ENV FILES_BASE=/usr/app/files
ENV YT_DLP_BIN=/usr/local/bin/yt-dlp
ENV VENV_PATH=/venv
ENV WHISPER_MODEL=small
ENV GPU_MODELS="tiny base small"
ENV SUMMARIZE=true
ENV OLLAMA_MODEL=qwen3:1.7b
ENV OLLAMA_URL=http://ollama:11434
ENV OLLAMA_CLOUD_URL=https://ollama.com/api
ENV OLLAMA_CLOUD_MODEL=gpt-oss:120b
ENV FORCE_LOCAL_SUMMARY=false
ENV FORCE_WHISPER=false
ENV CACHE_FILE_AGE_DAYS=30
ENV POLL_INTERVAL_MS=2000

# ─── Port ─────────────────────────────────────────────────────────────────────
EXPOSE 8501

# ─── Entrypoint ───────────────────────────────────────────────────────────────
CMD ["/venv/bin/streamlit", "run", "/usr/app/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
