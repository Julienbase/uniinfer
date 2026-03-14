# UniInfer

Hardware-agnostic AI inference runtime. Run any LLM on any hardware with one line of Python.

## Install

```bash
pip install -e .
```

For NVIDIA GPU support:
```bash
pip install -e ".[cuda]"
```

## Quick Start

```python
import uniinfer

# One-liner
text = uniinfer.generate("TheBloke/Llama-2-7B-GGUF", "What is gravity?")

# Engine for repeated use
engine = uniinfer.Engine(model="TheBloke/Llama-2-7B-GGUF")
result = engine.generate("Explain quantum computing.")
print(result.text)

# Streaming
for chunk in engine.stream("Write a haiku about code"):
    print(chunk.text, end="", flush=True)

# See your hardware
print(uniinfer.devices())
```

## CLI

```bash
# List available hardware
uniinfer devices

# Interactive chat
uniinfer chat --model TheBloke/Llama-2-7B-GGUF

# One-shot generation
uniinfer generate --model TheBloke/Llama-2-7B-GGUF --prompt "Hello world"

# Download a model
uniinfer pull --model TheBloke/Llama-2-7B-GGUF

# Benchmark
uniinfer bench --model TheBloke/Llama-2-7B-GGUF
```

## Supported Hardware

| Hardware | Status | Detection |
|----------|--------|-----------|
| NVIDIA CUDA | Supported | via pynvml |
| AMD ROCm | Supported | via rocm-smi |
| Vulkan | Supported | via vulkaninfo |
| CPU | Always available | via psutil |

Auto-detection picks the best available device. Override with `device="cuda:0"`, `device="cpu"`, etc.

## How It Works

UniInfer wraps llama.cpp (via llama-cpp-python) with a hardware abstraction layer that automatically:
1. Detects your hardware (GPU type, VRAM, CPU cores)
2. Downloads the right GGUF model from HuggingFace
3. Selects optimal quantization for your memory
4. Loads the model on the best available device

## License

MIT
