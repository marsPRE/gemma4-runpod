"""
handler.py — RunPod Serverless handler for Gemma 4 via llama-server.

Responsibilities:
  1. Ensure the GGUF model is downloaded (Network Volume cache).
  2. Launch llama-server as a subprocess.
  3. Wait until the server is healthy.
  4. Bridge incoming RunPod jobs to the llama-server OpenAI-compatible API.
  5. Support multimodal (image) inputs via the llama-server vision API.
  6. Support sync and streaming response modes.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Generator

import requests
import runpod
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("gemma4-handler")

# ---------------------------------------------------------------------------
# Configuration (all overridable via environment variables)
# ---------------------------------------------------------------------------
MODEL_REPO     = os.environ.get("MODEL_REPO",    "ggml-org/gemma-4-26B-A4B-it-GGUF")
MODEL_FILE     = os.environ.get("MODEL_FILE",    "gemma-4-26B-A4B-it-Q4_K_M.gguf")
MODEL_DIR      = os.environ.get("MODEL_DIR",     "/runpod-volume/models")
N_GPU_LAYERS   = os.environ.get("N_GPU_LAYERS",  "-1")
CTX_SIZE       = os.environ.get("CTX_SIZE",      "8192")
PARALLEL       = os.environ.get("PARALLEL",      "1")
LLAMA_PORT     = int(os.environ.get("LLAMA_PORT", "8080"))

MODEL_PATH     = Path(MODEL_DIR) / MODEL_FILE
LLAMA_BASE_URL = f"http://127.0.0.1:{LLAMA_PORT}"

_llama_proc: subprocess.Popen | None = None

# ---------------------------------------------------------------------------
# Step 1 – Model download
# ---------------------------------------------------------------------------

def ensure_model() -> None:
    """Download the GGUF model if it is not already cached."""
    if MODEL_PATH.exists():
        log.info("Model already cached at %s", MODEL_PATH)
        return

    log.info("Downloading model %s/%s → %s", MODEL_REPO, MODEL_FILE, MODEL_DIR)
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

    hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False,
    )
    log.info("Model download complete: %s", MODEL_PATH)


# ---------------------------------------------------------------------------
# Step 2 – Start llama-server
# ---------------------------------------------------------------------------

def start_llama_server() -> None:
    """Launch llama-server subprocess and wait until it is healthy."""
    global _llama_proc

    # Find llama-server binary — location varies by image version
    import shutil
    candidates = [
        "/llama-server",
        "/usr/local/bin/llama-server",
        "/app/llama-server",
        shutil.which("llama-server") or "",
        shutil.which("server") or "",
    ]
    binary = next((p for p in candidates if p and Path(p).exists()), None)
    if binary is None:
        # Log all files to help debug
        import glob
        found = glob.glob("/*server*") + glob.glob("/usr/**/*server*", recursive=True)
        log.error("llama-server not found! Located files: %s", found)
        raise FileNotFoundError("llama-server binary not found. Searched: " + str(candidates))
    log.info("Using llama-server binary: %s", binary)

    cmd = [
        binary,
        "--model",       str(MODEL_PATH),
        "--host",        "127.0.0.1",
        "--port",        str(LLAMA_PORT),
        "--n-gpu-layers", N_GPU_LAYERS,
        "--ctx-size",    CTX_SIZE,
        "--parallel",    PARALLEL,
        # Enable vision / multimodal support if the model ships a mmproj file
        # (llama-server detects it automatically when placed next to the GGUF)
    ]

    # If a multimodal projector file exists next to the GGUF, pass it explicitly.
    mmproj_candidates = list(Path(MODEL_DIR).glob("*mmproj*"))
    if mmproj_candidates:
        cmd += ["--mmproj", str(mmproj_candidates[0])]
        log.info("Multimodal projector found: %s", mmproj_candidates[0])

    log.info("Starting llama-server: %s", " ".join(cmd))
    _llama_proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)

    # Poll health endpoint
    health_url = f"{LLAMA_BASE_URL}/health"
    deadline    = time.time() + 300  # 5-minute timeout
    while time.time() < deadline:
        try:
            r = requests.get(health_url, timeout=5)
            if r.status_code == 200 and r.json().get("status") == "ok":
                log.info("llama-server is healthy.")
                return
        except Exception:
            pass
        time.sleep(2)

    raise RuntimeError("llama-server did not become healthy within 300 s")


# ---------------------------------------------------------------------------
# Helper – build message list with optional image
# ---------------------------------------------------------------------------

def _build_messages(job_input: dict) -> list[dict]:
    """
    Convert job input to OpenAI-style message list.

    Supported input shapes:
      • {"messages": [...]}          – pass-through
      • {"prompt": "...", "image": "<base64>"}  – user text + image
      • {"prompt": "..."}            – plain text
    """
    if "messages" in job_input:
        return job_input["messages"]

    prompt = job_input.get("prompt", "")
    image  = job_input.get("image")  # base64-encoded image string or data-URL

    if image:
        # Normalise to data-URL if raw base64 was supplied
        if not image.startswith("data:"):
            image = f"data:image/jpeg;base64,{image}"

        content = [
            {"type": "image_url", "image_url": {"url": image}},
            {"type": "text",      "text": prompt},
        ]
    else:
        content = prompt

    return [{"role": "user", "content": content}]


# ---------------------------------------------------------------------------
# Step 3 – Sync handler
# ---------------------------------------------------------------------------

def handler(job: dict) -> dict | Generator:
    """RunPod job handler – sync and streaming."""
    job_input = job.get("input", {})

    # RunPod OpenAI-proxy mode: requests via /openai/v1/... are forwarded here
    # with openai_route + openai_input set by the RunPod gateway.
    if "openai_route" in job_input:
        route  = job_input["openai_route"].lstrip("/")   # e.g. "v1/chat/completions"
        body   = job_input.get("openai_input", {})
        stream = body.get("stream", False)
        url    = f"{LLAMA_BASE_URL}/{route}"
        if stream:
            return _stream_openai(url, body)
        resp = requests.post(url, json=body, timeout=300)
        resp.raise_for_status()
        result = resp.json()
        # Strip non-standard llama.cpp fields that may confuse RunPod's proxy
        result.pop("timings", None)
        result.pop("system_fingerprint", None)
        # Gemma 4 thinking-mode outputs reasoning_content instead of content.
        # Move reasoning_content into content so clients receive actual text,
        # and strip the non-standard field to keep the response OpenAI-compatible.
        for choice in result.get("choices", []):
            msg = choice.get("message", {})
            reasoning = msg.pop("reasoning_content", None)
            if reasoning and not msg.get("content"):
                msg["content"] = reasoning
        return result

    stream  = job_input.get("stream", False)
    payload = {
        "model":       MODEL_FILE,
        "messages":    _build_messages(job_input),
        "max_tokens":  job_input.get("max_tokens",  512),
        "temperature": job_input.get("temperature", 0.7),
        "top_p":       job_input.get("top_p",       0.9),
        "stream":      stream,
    }

    if stream:
        return _stream_response(payload)
    return _sync_response(payload)


def _sync_response(payload: dict) -> dict:
    url = f"{LLAMA_BASE_URL}/v1/chat/completions"
    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()

    # Extract text content
    text = ""
    choices = data.get("choices", [])
    if choices:
        text = choices[0].get("message", {}).get("content", "")

    return {
        "response":    text,
        "usage":       data.get("usage", {}),
        "model":       data.get("model", MODEL_FILE),
        "finish_reason": choices[0].get("finish_reason", "") if choices else "",
    }


def _stream_openai(url: str, body: dict) -> Generator:
    """Pass SSE lines from llama-server straight through to the RunPod gateway."""
    with requests.post(url, json=body, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if raw_line:
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                yield line + "\n"


def _stream_response(payload: dict) -> Generator:
    url = f"{LLAMA_BASE_URL}/v1/chat/completions"
    with requests.post(url, json=payload, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ensure_model()
    start_llama_server()

    log.info("Registering RunPod handler...")
    runpod.serverless.start({"handler": handler})
