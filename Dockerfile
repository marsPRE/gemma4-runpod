# ============================================================
# Stage 1: Build llama.cpp with CUDA support
# ============================================================
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp .

RUN cmake -B build \
        -DGGML_CUDA=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_CURL=ON \
    && cmake --build build --config Release -j"$(nproc)" --target llama-server

# ============================================================
# Stage 2: Runtime image
# ============================================================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        libcurl4 \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy compiled binary from builder
COPY --from=builder /build/build/bin/llama-server /usr/local/bin/llama-server
RUN chmod +x /usr/local/bin/llama-server

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application files
COPY handler.py  /app/handler.py
COPY startup.sh  /app/startup.sh
RUN chmod +x /app/startup.sh

# ---- Environment defaults ----
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV HF_HUB_DISABLE_XET=1
ENV MODEL_REPO=ggml-org/gemma-4-26B-A4B-it-GGUF
ENV MODEL_FILE=gemma-4-26B-A4B-it-Q4_K_M.gguf
ENV MODEL_DIR=/runpod-volume/models
ENV N_GPU_LAYERS=-1
ENV CTX_SIZE=8192
ENV PARALLEL=1
ENV LLAMA_PORT=8080

CMD ["/app/startup.sh"]
