# UniInfer

**Hardware-agnostic AI inference runtime.** Run any LLM on any hardware — NVIDIA, AMD, Intel, or CPU — with zero configuration.

*"Ollama makes it easy to run models. UniInfer makes it impossible to run them wrong."*

```python
import uniinfer

response = uniinfer.chat("mistral-7b", "What is gravity?")
print(response)
```

No CUDA setup. No driver matching. No format conversion. It just works.

---

## Why UniInfer?

Running local AI models today means navigating a maze of hardware drivers, model formats, and cryptic OOM crashes. Download a 37GB model that won't fit your GPU? Nobody tells you until it fails. Switch GPUs? Everything breaks.

UniInfer sits between your code and the hardware complexity:

```
Your code (Python SDK / REST API / CLI)
              |
          UniInfer
     Smart Fitting + Fallback + Diagnostics
              |
  +-----------+-----------+---------+--------+
  | NVIDIA    |    AMD    | Vulkan  |  CPU   |
  | (CUDA)    |  (ROCm)  |         |        |
  +-----------+-----------+---------+--------+
```

- **Smart Fitting** — knows what fits your VRAM *before* downloading
- **Auto-detects** your hardware (GPU type, VRAM, compute capability)
- **Auto-downloads** the right model format from HuggingFace
- **Auto-selects** optimal quantization for your available memory
- **Auto-falls back** when a device fails (CUDA → ROCm → Vulkan → CPU)
- **Multi-format** — GGUF, ONNX, SafeTensors — drop in any format

## Installation

```bash
pip install -e .
```

For NVIDIA GPU acceleration (requires CUDA toolkit):

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

## Quick Start

### One-Line SDK

```python
import uniinfer

# Simplest possible usage — alias resolves to full HuggingFace repo
response = uniinfer.chat("mistral-7b", "What is gravity?")
print(response)

# Stream tokens
for chunk in uniinfer.chat_stream("mistral-7b", "Write a poem about code."):
    print(chunk, end="", flush=True)

# With a system prompt
response = uniinfer.chat("mistral-7b", "Explain quantum computing", system="You are a physicist.")
```

### Engine API (Full Control)

```python
from uniinfer import Engine

engine = Engine(model="mistral-7b")

# Chat with proper templates
result = engine.chat([
    {"role": "user", "content": "Explain quantum computing in simple terms."}
])
print(result.text)

# Multi-turn conversation
result = engine.chat([
    {"role": "user", "content": "My name is Julien."},
    {"role": "assistant", "content": "Nice to meet you, Julien!"},
    {"role": "user", "content": "What is my name?"},
])
print(result.text)  # "Your name is Julien."

# Stream tokens in real-time
for chunk in engine.chat_stream([
    {"role": "user", "content": "Write a haiku about code."}
]):
    print(chunk.text, end="", flush=True)

# Raw text generation
result = engine.generate("The history of computing began", max_tokens=200)

# Force a specific device
engine_cpu = Engine(model="mistral-7b", device="cpu")
engine_gpu = Engine(model="mistral-7b", device="cuda:0")

# Diagnostics — see tok/s, memory usage, fallback events
info = engine.info()
print(info["diagnostics"])  # {"average_tokens_per_second": 56.8, ...}
print(info["fit"])           # {"fits": true, "headroom_gb": 7.9, ...}
```

### REST API (OpenAI-Compatible)

```bash
# Start the server with a model
uniinfer serve --model mistral-7b --port 8000

# Or start without a model — load later from the dashboard
uniinfer serve

# Use with any OpenAI-compatible client
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mistral-7b", "messages": [{"role": "user", "content": "Hello"}]}'
```

### Web Dashboard

Start the server and open `http://localhost:8000/dashboard` in your browser:

- **Dashboard** — live system status, VRAM gauge, throughput chart, queue depth
- **Chat** — interactive chat playground with model switching, streaming, session history
- **Generate** — one-shot text generation with output stats
- **Models** — download, load, delete models; browse aliases; hot-swap models without restart
- **Devices** — hardware device listing with memory stats
- **Bench** — configurable inference benchmark with per-run results
- **Fit Check** — check if a model fits your hardware before downloading

### CLI

```bash
# See what hardware UniInfer detects
uniinfer devices

# Interactive chat session (use aliases or full repo IDs)
uniinfer chat --model mistral-7b
uniinfer chat --model "TheBloke/Mistral-7B-Instruct-v0.2-GGUF" --device cuda:0

# Connect to a running server (metrics appear in dashboard)
uniinfer chat --server

# One-shot text generation
uniinfer generate --model mistral-7b --prompt "Hello world"

# Check if a model fits your hardware before loading
uniinfer run --model llama-3.1-8b --dry-run

# List available model aliases
uniinfer aliases

# Download a model ahead of time
uniinfer pull --model "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"

# List cached models
uniinfer list

# Benchmark tokens/sec
uniinfer bench --model mistral-7b
```

## Smart Model Fitting

UniInfer checks if a model fits your hardware *before* downloading — no more wasted bandwidth or OOM crashes.

```bash
$ uniinfer run --model llama-3.1-8b --dry-run

UniInfer — Smart Model Fit Check

Discovering hardware... NVIDIA GeForce RTX 3060 (11.2 GB free)
Model: Llama 3.1 8B Instruct (8.0B params)

┌─────── Fit Report ───────┐
│ Status:    FITS           │
│ Model:     ~4.5 GB (q4_k_m) │
│ Available: 11.2 GB        │
│ Overhead:  0.5 GB         │
│ Headroom:  +4.0 GB        │
└──────────────────────────┘

┌─── Quantization Options ───┐
│ Quant   Est. Size  Fits?   │
│ f16     16.0 GB    No      │
│ q8_0    8.0 GB     Yes     │
│ q5_k_m  5.5 GB     Yes     │
│ q4_k_m  4.5 GB     Yes     │
│ q3_k_m  3.1 GB     Yes     │
│ q2_k    2.4 GB     Yes     │
└────────────────────────────┘
```

If a model doesn't fit, UniInfer recommends a smaller quantization or alternative model — instead of crashing with a cryptic error.

## Hardware Resilience

If your preferred GPU fails (driver mismatch, CUDA error, etc.), UniInfer automatically falls back:

```
CUDA → ROCm → Vulkan → CPU
```

Each device is health-checked before attempting to load. Fallback events are logged and exposed via `engine.info()["fallback"]`.

## Model Aliases

Use short names instead of full HuggingFace repo IDs:

| Alias | Model | Params | Default Quant |
|-------|-------|--------|---------------|
| `tinyllama-1b` | TinyLlama 1.1B Chat | 1.1B | q4_k_m |
| `gemma-2b` | Gemma 2B Instruct | 2.5B | q4_k_m |
| `phi-3-mini` | Phi-3 Mini 3.8B | 3.8B | q4_k_m |
| `mistral-7b` | Mistral 7B Instruct v0.2 | 7.2B | q4_k_m |
| `llama-3.1-8b` | Llama 3.1 8B Instruct | 8.0B | q4_k_m |
| `qwen-2.5-7b` | Qwen 2.5 7B Instruct | 7.6B | q4_k_m |
| `llama-3.3-70b` | Llama 3.3 70B Instruct | 70.6B | q4_k_m |

Run `uniinfer aliases` to see the full list.

## Supported Hardware

| Hardware | Status | Backend | Detection |
|----------|--------|---------|-----------|
| NVIDIA GPUs (CUDA) | Supported | llama.cpp, transformers | via pynvml |
| AMD GPUs (ROCm) | Supported | llama.cpp, transformers | via rocm-smi |
| Vulkan GPUs | Supported | llama.cpp | via vulkaninfo |
| CPU (x86/ARM) | Always available | llama.cpp, ONNX Runtime, transformers | via psutil |

## Supported Model Formats

| Format | Backend | Auto-Detected By |
|--------|---------|------------------|
| GGUF | llama.cpp | File extension + magic bytes |
| ONNX | ONNX Runtime | File extension |
| SafeTensors / HuggingFace | transformers | Directory with `.safetensors` files |

Drop in any supported format — UniInfer auto-detects and routes to the right backend.

## Benchmarks

Tested on RTX 3060 (12GB) + AMD Ryzen, Mistral 7B Q4_K_M:

| Device | Speed | Relative |
|--------|-------|----------|
| **GPU (CUDA)** | **56.8 tok/s** | 6.9x faster |
| CPU (Ryzen) | 8.3 tok/s | baseline |

Same model, same API, same command — just a `--device` flag change.

## How It Works

1. **Hardware Discovery** — Probes CUDA, ROCm, Vulkan, and CPU. Ranks devices by type and available memory. Health-checks each device before use.
2. **Smart Fitting** — Calculates VRAM budget (model + KV cache + overhead) and validates the model fits before downloading. Recommends quantization alternatives if it doesn't.
3. **Model Resolution** — Resolves aliases (e.g., `mistral-7b`), checks local cache, auto-detects format (GGUF/ONNX/SafeTensors), downloads from HuggingFace.
4. **Hardware Resilience** — If the preferred device fails, automatically falls back through the chain (CUDA → ROCm → Vulkan → CPU) with clear messaging.
5. **Backend Loading** — Routes to the right backend (llama.cpp, ONNX Runtime, or transformers) with correct GPU offloading, context size, and thread count.
6. **Diagnostics** — Every inference call is timed. `engine.info()` exposes tok/s, load time, memory headroom, and fallback events.

## Architecture

```
src/uniinfer/
  hal/            # Hardware Abstraction Layer
    discovery.py    # Auto-detect all available hardware
    interface.py    # DeviceInfo dataclass + DeviceType enum
    health.py       # Device health checks (CUDA/ROCm/Vulkan/CPU)
    cuda_adapter.py # NVIDIA GPU detection
    rocm_adapter.py # AMD GPU detection
    vulkan_adapter.py # Vulkan GPU detection
    cpu_adapter.py  # CPU detection

  backends/       # Execution Backends
    interface.py    # Abstract ExecutionBackend class
    registry.py     # Backend detection (magic bytes, extensions, directories)
    llamacpp.py     # llama-cpp-python backend
    onnxrt.py       # ONNX Runtime backend
    transformers_backend.py  # HuggingFace transformers backend

  models/         # Model Management
    registry.py     # HuggingFace download + caching + pre-download fit check
    aliases.py      # Model alias registry (e.g., "mistral-7b")
    fitting.py      # VRAM budget calculator + FitReport
    gguf_metadata.py # GGUF binary header parser
    quantization.py # Smart quantization selection + size estimation
    converter.py    # Device-to-quantization mapping

  engine/         # Core Engine
    engine.py       # Main user-facing Engine class
    fallback.py     # Hardware fallback chain (CUDA -> ROCm -> Vulkan -> CPU)
    diagnostics.py  # Inference timing + session metrics
    scheduler.py    # Async request scheduler

  api/            # REST API + Dashboard
    server.py       # FastAPI app + model hot-swap
    routes_dashboard.py  # Dashboard REST + SSE endpoints
    chat_store.py   # In-memory chat session store
    schemas.py      # Pydantic request/response models
    static/         # Built React dashboard assets

  cli/            # Command Line Interface
    main.py         # Typer CLI (chat, generate, run, serve, aliases, bench)

web/              # React Dashboard (Vite + React 19 + TypeScript + Tailwind v4)
  src/pages/        # Dashboard, Chat, Generate, Models, Devices, Bench, FitCheck
  src/api/          # API client, hooks, types
  src/components/   # Reusable UI components
```

## Roadmap

- **v0.1** ✅ — Hardware abstraction, llama.cpp backend, auto-download, auto-quantization, Python SDK + CLI
- **v0.5** ✅ — OpenAI-compatible REST API, async scheduler, SSE streaming, ONNX Runtime backend, Prometheus metrics
- **v1.0** ✅ — Smart model fitting, multi-format loading, hardware resilience, inference diagnostics, one-line SDK + model aliases
- **v1.5** ✅ — Web dashboard, interactive chat, model hot-swap, server-connected CLI, modelless server start
- **v2.0** — Enterprise tier (RBAC, audit logs, SSO, air-gapped deployment)

See [docs/ROADMAP.md](docs/ROADMAP.md) for the detailed plan and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for technical design.

## Contributing

UniInfer is open source under the MIT license. Contributions welcome.

## License

MIT
