"""
client.py — Python client for the gemma4-runpod RunPod Serverless endpoint.

Supports:
  • Synchronous requests (blocking)
  • Asynchronous requests (submit + poll)
  • Streaming responses (chunked SSE)
  • Image inputs (base64 / file path / data-URL)

Usage:
  python client.py "What is the capital of Germany?"
  python client.py --stream "Tell me a story"
  python client.py --image /path/to/photo.jpg "Describe this image"
  python client.py --async-mode "Explain relativity"
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_KEY     = os.environ.get("RUNPOD_API_KEY", "")
ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "")
BASE_URL    = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type":  "application/json",
}

# ---------------------------------------------------------------------------
# Helper – encode image
# ---------------------------------------------------------------------------

def _encode_image(source: str) -> str:
    """
    Return a base64 data-URL from:
      • a local file path
      • an existing data-URL (returned as-is)
      • raw base64 string (wrapped in data-URL)
    """
    if source.startswith("data:"):
        return source

    path = Path(source)
    if path.exists():
        suffix   = path.suffix.lower().lstrip(".")
        mime     = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                    "png": "image/png",  "gif": "image/gif",
                    "webp": "image/webp"}.get(suffix, "image/jpeg")
        b64_data = base64.b64encode(path.read_bytes()).decode()
        return f"data:{mime};base64,{b64_data}"

    # Assume raw base64
    return f"data:image/jpeg;base64,{source}"


# ---------------------------------------------------------------------------
# Sync request
# ---------------------------------------------------------------------------

def run_sync(
    prompt: str,
    image: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """Submit a job and block until completion."""
    payload: dict = {
        "input": {
            "prompt":      prompt,
            "max_tokens":  max_tokens,
            "temperature": temperature,
            "stream":      False,
        }
    }
    if image:
        payload["input"]["image"] = _encode_image(image)

    resp = requests.post(f"{BASE_URL}/runsync", headers=HEADERS, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    return data.get("output", {}).get("response", str(data))


# ---------------------------------------------------------------------------
# Async request
# ---------------------------------------------------------------------------

def run_async(
    prompt: str,
    image: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    poll_interval: float = 2.0,
    timeout: float = 300.0,
) -> str:
    """Submit a job and poll until completion."""
    payload: dict = {
        "input": {
            "prompt":      prompt,
            "max_tokens":  max_tokens,
            "temperature": temperature,
            "stream":      False,
        }
    }
    if image:
        payload["input"]["image"] = _encode_image(image)

    # Submit
    resp = requests.post(f"{BASE_URL}/run", headers=HEADERS, json=payload, timeout=60)
    resp.raise_for_status()
    job_id = resp.json()["id"]
    print(f"[async] Job submitted: {job_id}", file=sys.stderr)

    # Poll
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(poll_interval)
        status_resp = requests.get(f"{BASE_URL}/status/{job_id}", headers=HEADERS, timeout=30)
        status_resp.raise_for_status()
        status_data = status_resp.json()
        status = status_data.get("status", "")
        print(f"[async] Status: {status}", file=sys.stderr)

        if status == "COMPLETED":
            return status_data.get("output", {}).get("response", str(status_data))
        if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
            raise RuntimeError(f"Job {job_id} ended with status: {status}")

    raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")


# ---------------------------------------------------------------------------
# Streaming request
# ---------------------------------------------------------------------------

def run_stream(
    prompt: str,
    image: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> None:
    """Submit a streaming job and print chunks as they arrive."""
    payload: dict = {
        "input": {
            "prompt":      prompt,
            "max_tokens":  max_tokens,
            "temperature": temperature,
            "stream":      True,
        }
    }
    if image:
        payload["input"]["image"] = _encode_image(image)

    # Submit
    resp = requests.post(f"{BASE_URL}/run", headers=HEADERS, json=payload, timeout=60)
    resp.raise_for_status()
    job_id = resp.json()["id"]
    print(f"[stream] Job submitted: {job_id}", file=sys.stderr)

    # Stream results
    stream_url = f"{BASE_URL}/stream/{job_id}"
    deadline   = time.time() + 300
    seen       = 0

    while time.time() < deadline:
        time.sleep(0.5)
        sr = requests.get(stream_url, headers=HEADERS, timeout=30)
        sr.raise_for_status()
        stream_data = sr.json()

        chunks = stream_data.get("stream", [])
        for chunk in chunks[seen:]:
            output = chunk.get("output", "")
            if output:
                print(output, end="", flush=True)
        seen = len(chunks)

        if stream_data.get("status") in ("COMPLETED", "FAILED", "CANCELLED"):
            print()  # Final newline
            break


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="gemma4-runpod client — sync / async / streaming with optional image input"
    )
    parser.add_argument("prompt", help="Text prompt to send to the model")
    parser.add_argument("--stream",     action="store_true", help="Use streaming mode")
    parser.add_argument("--async-mode", action="store_true", dest="async_mode",
                        help="Use async (poll) mode instead of runsync")
    parser.add_argument("--image",      default=None,
                        help="Path to an image file or base64 string for multimodal input")
    parser.add_argument("--max-tokens", type=int,   default=512,  dest="max_tokens")
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()


def main() -> None:
    if not API_KEY:
        sys.exit("Error: RUNPOD_API_KEY environment variable is not set.")
    if not ENDPOINT_ID:
        sys.exit("Error: RUNPOD_ENDPOINT_ID environment variable is not set.")

    args = _parse_args()

    if args.stream:
        run_stream(args.prompt, image=args.image,
                   max_tokens=args.max_tokens, temperature=args.temperature)
    elif args.async_mode:
        result = run_async(args.prompt, image=args.image,
                           max_tokens=args.max_tokens, temperature=args.temperature)
        print(result)
    else:
        result = run_sync(args.prompt, image=args.image,
                          max_tokens=args.max_tokens, temperature=args.temperature)
        print(result)


if __name__ == "__main__":
    main()
