# ============================================================
# Base: pre-built llama.cpp server with CUDA (maintained by ggml-org)
# Avoids compiling CUDA on CI runners that have no GPU driver.
# ============================================================
FROM ghcr.io/ggml-org/llama.cpp:server-cuda

ENTRYPOINT []

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV LD_LIBRARY_PATH="/app:${LD_LIBRARY_PATH:-}"

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# Copy application files
COPY handler.py  /app/handler.py
COPY startup.sh  /app/startup.sh
RUN chmod +x /app/startup.sh

# ---- Environment defaults ----
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV HF_HUB_DISABLE_XET=1
ENV MODEL_REPO=LiconStudio/Gemma-4-31B-it-abliterated-GGUF
ENV MODEL_FILE=gemma-4-31B-it-abliterated-Q4_K_M.gguf
ENV MODEL_DIR=/runpod-volume/models
ENV N_GPU_LAYERS=-1
ENV CTX_SIZE=8192
ENV PARALLEL=1
ENV LLAMA_PORT=8080

CMD ["/app/startup.sh"]
