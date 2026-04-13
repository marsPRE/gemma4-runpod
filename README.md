# gemma4-runpod

> RunPod Serverless worker — Gemma 4 26B-A4B-it via llama.cpp (CUDA) with OpenAI-compatible API

---

## Deutsch

### Überblick

Dieses Projekt stellt Google Gemma 4 26B-A4B-it als RunPod Serverless-Endpunkt bereit.  
Die Inferenz läuft über `llama-server` mit CUDA-Beschleunigung.  
Das Modell wird beim ersten Start automatisch vom HuggingFace Hub heruntergeladen und auf dem Network Volume gecacht.

### Features

- Multi-Stage Docker-Build (kompiliert llama.cpp mit CUDA aus dem Quellcode)
- Automatischer Modell-Download + Network-Volume-Cache
- OpenAI-kompatibler API (`/v1/chat/completions`)
- Multimodales Eingabe-Support (Bilder als base64)
- Python-Client mit sync, async und Streaming

### Deployment auf RunPod

1. **Repository forken** und GitHub Actions aktivieren
2. **GHCR-Paket öffentlich stellen** (Package Settings → Change Visibility)
3. **RunPod Serverless Endpunkt erstellen:**
   - Container Image: `ghcr.io/<dein-user>/gemma4-runpod:latest`
   - GPU: RTX 4090 oder L4 (min. 24 GB VRAM)
   - Min Workers: 0 | Max Workers: 2
   - Idle Timeout: 30 s
   - Network Volume mounten auf `/runpod-volume`
4. **Umgebungsvariablen setzen** (siehe `.env.example`)

### Nutzung

```bash
export RUNPOD_API_KEY=dein_api_key
export RUNPOD_ENDPOINT_ID=dein_endpoint_id

# Synchron
python client.py "Was ist die Hauptstadt Frankreichs?"

# Streaming
python client.py --stream "Schreibe ein Gedicht über den Herbst"

# Mit Bild (multimodal)
python client.py --image /pfad/zum/bild.jpg "Was siehst du auf diesem Bild?"
```

---

## English

### Overview

This project deploys Google Gemma 4 26B-A4B-it as a RunPod Serverless endpoint.  
Inference runs via `llama-server` with CUDA acceleration.  
The model is automatically downloaded from HuggingFace Hub on first startup and cached on the Network Volume.

### Features

- Multi-stage Docker build (compiles llama.cpp with CUDA from source)
- Automatic model download + Network Volume caching
- OpenAI-compatible API (`/v1/chat/completions`)
- Multimodal input support (images as base64)
- Python client with sync, async, and streaming modes

### Environment Variables

| Variable        | Default                                | Description                        |
|-----------------|----------------------------------------|------------------------------------|
| `MODEL_REPO`    | `ggml-org/gemma-4-26B-A4B-it-GGUF`    | HuggingFace repository             |
| `MODEL_FILE`    | `gemma-4-26B-A4B-it-Q4_K_M.gguf`      | GGUF model filename                |
| `MODEL_DIR`     | `/runpod-volume/models`                | Model cache directory              |
| `N_GPU_LAYERS`  | `-1`                                   | Number of layers on GPU (-1 = all) |
| `CTX_SIZE`      | `8192`                                 | Context window size (tokens)       |
| `PARALLEL`      | `1`                                    | Parallel request slots             |
| `LLAMA_PORT`    | `8080`                                 | Internal llama-server port         |

### Deployment on RunPod

1. **Fork this repository** and enable GitHub Actions
2. **Make the GHCR package public** (Package Settings → Change Visibility)
3. **Create RunPod Serverless Endpoint:**
   - Container Image: `ghcr.io/<your-user>/gemma4-runpod:latest`
   - GPU: RTX 4090 or L4 (min. 24 GB VRAM)
   - Min Workers: 0 | Max Workers: 2
   - Idle Timeout: 30 s
   - Mount Network Volume at `/runpod-volume`
4. **Set environment variables** (see `.env.example`)

### Usage

```bash
export RUNPOD_API_KEY=your_api_key
export RUNPOD_ENDPOINT_ID=your_endpoint_id

# Synchronous
python client.py "What is the capital of France?"

# Streaming
python client.py --stream "Write a poem about autumn"

# With image (multimodal)
python client.py --image /path/to/image.jpg "Describe this image"

# Async mode
python client.py --async-mode "Explain quantum entanglement"
```

### Build Locally

```bash
docker build -t gemma4-runpod .
```

### License

MIT
