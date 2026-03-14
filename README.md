# UniInfer

**Hardware-agnostic AI inference runtime.** Run any LLM on any hardware — NVIDIA, AMD, Intel, or CPU — with zero configuration.

One API. One command. Any hardware.

```python
from uniinfer import Engine

engine = Engine(model="TheBloke/Mistral-7B-Instruct-v0.2-GGUF")
result = engine.chat([{"role": "user", "content": "What is gravity?"}])
print(result.text)
# "Gravity is a fundamental force of nature..."
```

No CUDA setup. No driver matching. No format conversion. It just works.

---

## Why UniInfer?

Today, running AI models means being locked to NVIDIA's CUDA ecosystem. Switching hardware means rewriting code, debugging drivers, and losing weeks.

UniInfer fixes that by sitting between your code and the hardware:

```
Your code (Python / CLI)
        |
    UniInfer
        |
  +-----------+-----------+---------+
  | NVIDIA    |    AMD    |  Intel  |  CPU
  | (CUDA)    |  (ROCm)  | (oneAPI)|
  +-----------+-----------+---------+
```

- **Auto-detects** your hardware (GPU type, VRAM, compute capability)
- **Auto-downloads** the right model format from HuggingFace
- **Auto-selects** optimal quantization for your available memory
- **Auto-routes** computation to the best available device

## Installation

```bash
pip install -e .
```

For NVIDIA GPU acceleration (requires CUDA toolkit):

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

## Quick Start

### Python API

```python
from uniinfer import Engine

# Auto-detect hardware, download model, run inference
engine = Engine(model="TheBloke/Mistral-7B-Instruct-v0.2-GGUF")

# Chat with proper templates (reads format from model metadata)
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
    {"role": "user", "content": "Write a poem about code."}
]):
    print(chunk.text, end="", flush=True)

# Raw text generation
result = engine.generate("The history of computing began", max_tokens=200)

# Force a specific device
engine_cpu = Engine(model="TheBloke/Mistral-7B-Instruct-v0.2-GGUF", device="cpu")
engine_gpu = Engine(model="TheBloke/Mistral-7B-Instruct-v0.2-GGUF", device="cuda:0")
```

### CLI

```bash
# See what hardware UniInfer detects
uniinfer devices

# Interactive chat session
uniinfer chat --model "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"

# Force CPU or GPU
uniinfer chat --model "TheBloke/Mistral-7B-Instruct-v0.2-GGUF" --device cpu
uniinfer chat --model "TheBloke/Mistral-7B-Instruct-v0.2-GGUF" --device cuda:0

# One-shot text generation
uniinfer generate --model "TheBloke/Mistral-7B-Instruct-v0.2-GGUF" --prompt "Hello world"

# Download a model ahead of time
uniinfer pull --model "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"

# List cached models
uniinfer list

# Benchmark tokens/sec
uniinfer bench --model "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
```

## Supported Hardware

| Hardware | Status | Backend | Detection |
|----------|--------|---------|-----------|
| NVIDIA GPUs (CUDA) | Supported | llama.cpp | via pynvml |
| AMD GPUs (ROCm) | Supported | llama.cpp | via rocm-smi |
| Vulkan GPUs | Supported | llama.cpp | via vulkaninfo |
| CPU (x86/ARM) | Always available | llama.cpp | via psutil |

Auto-detection picks the fastest available device. Override with `--device cuda:0`, `--device cpu`, etc.

## Benchmarks

Tested on RTX 3060 (12GB) + AMD Ryzen, Mistral 7B Q4_K_M:

| Device | Speed | Relative |
|--------|-------|----------|
| **GPU (CUDA)** | **56.8 tok/s** | 6.9x faster |
| CPU (Ryzen) | 8.3 tok/s | baseline |

Same model, same API, same command — just a `--device` flag change.

## Architecture

```
src/uniinfer/
  hal/            # Hardware Abstraction Layer
    discovery.py    # Auto-detect all available hardware
    interface.py    # DeviceInfo dataclass + DeviceType enum
    cuda_adapter.py # NVIDIA GPU detection
    rocm_adapter.py # AMD GPU detection
    vulkan_adapter.py # Vulkan GPU detection
    cpu_adapter.py  # CPU detection

  backends/       # Execution Backends
    interface.py    # Abstract ExecutionBackend class
    llamacpp.py     # llama-cpp-python backend

  models/         # Model Management
    registry.py     # HuggingFace download + caching
    quantization.py # Smart quantization selection
    converter.py    # Device-to-quantization mapping

  engine/         # Core Engine
    engine.py       # Main user-facing Engine class
    sampling.py     # Sampling parameters

  config/         # Configuration
    engine_config.py # Validated engine settings

  cli/            # Command Line Interface
    main.py         # Typer CLI with 6 commands
```

## How It Works

1. **Hardware Discovery** — Probes CUDA, ROCm, Vulkan, and CPU. Ranks devices by type and available memory.
2. **Model Resolution** — Checks local cache, then downloads the right GGUF file from HuggingFace.
3. **Quantization Selection** — Based on available VRAM/RAM, picks the best quantization (F16 → Q8 → Q4_K_M) automatically.
4. **Chat Templates** — Reads the model's built-in chat template from GGUF metadata. Supports Mistral, Llama, ChatML, and any template the model defines.
5. **Backend Loading** — Loads via llama.cpp with correct GPU layer offloading, context size, and thread count.

## The Vision

UniInfer is the first step toward breaking AI's hardware lock-in. Today it handles inference. The roadmap:

- **v0.1** (current) — Single-model inference on CUDA/ROCm/CPU via llama.cpp
- **v0.5** — OpenAI-compatible REST API server, continuous batching
- **v1.0** — Multi-backend support (vLLM, TensorRT, ONNX Runtime)
- **v2.0** — Training abstraction layer

See [docs/RESEARCH.md](docs/RESEARCH.md) for the full landscape analysis and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for technical design.

## Contributing

UniInfer is open source under the MIT license. Contributions welcome.

## License

MIT
