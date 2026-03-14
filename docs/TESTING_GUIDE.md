# UniInfer Testing Guide

## Your Hardware

- **GPU**: NVIDIA GeForce RTX 3060 (12GB VRAM, CUDA compute capability 8.6)
- **CPU**: AMD Ryzen (8 cores, 16 threads, 16GB RAM)

---

## Setup

UniInfer is already installed. The CLI is located at:

```
C:\Users\Iulian\AppData\Roaming\Python\Python312\Scripts\uniinfer.exe
```

To make things easier, add the Scripts folder to your PATH:

```
set PATH=%PATH%;C:\Users\Iulian\AppData\Roaming\Python\Python312\Scripts
```

After that you can just type `uniinfer` instead of the full path.

---

## Step 1: Verify Hardware Detection

```bash
uniinfer devices
```

You should see your RTX 3060 (CUDA), Vulkan, and CPU listed.

---

## Step 2: Pull a Model

### Small model (TinyLlama 1.1B — fast, good for testing)

```bash
uniinfer pull --model "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
```

### Medium model (Mistral 7B — better quality, fits in 12GB VRAM with Q4)

```bash
uniinfer pull --model "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
```

### Larger model (Llama 3 8B — good quality, fits in 12GB VRAM with Q4)

```bash
uniinfer pull --model "QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF"
```

---

## Step 3: Test on GPU (CUDA — RTX 3060)

### One-shot generation

```bash
uniinfer generate --model "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF" --prompt "What is gravity? Answer in 2 sentences." --max-tokens 100
```

### Interactive chat

```bash
uniinfer chat --model "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
```

Type messages and press Enter. Type `exit` or `quit` to stop.

### Force GPU explicitly

```bash
uniinfer generate --model "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF" --prompt "Hello world" --device cuda:0
```

---

## Step 4: Test on CPU (AMD Ryzen)

### Force CPU mode

```bash
uniinfer generate --model "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF" --prompt "What is gravity? Answer in 2 sentences." --max-tokens 100 --device cpu
```

### Interactive chat on CPU

```bash
uniinfer chat --model "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF" --device cpu
```

CPU will be slower than GPU. This is expected. The point is that the same model, same command, same API — just a different device flag.

---

## Step 5: Benchmark GPU vs CPU

### Benchmark on GPU (auto-selects CUDA)

```bash
uniinfer bench --model "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
```

### Benchmark on CPU

```bash
uniinfer bench --model "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF" --device cpu
```

Compare tokens/sec between GPU and CPU. You should see GPU is significantly faster.

---

## Step 6: Test via Python

Create a file `test_uniinfer.py` anywhere and run it with Python 3.12:

```python
import uniinfer

# Check available hardware
print("=== Available Devices ===")
for device in uniinfer.devices():
    print(f"  {device.device_type.value}:{device.device_id} — {device.name} ({device.total_memory / 1e9:.1f} GB)")

# Generate on GPU (auto-detected)
print("\n=== GPU Generation ===")
response = uniinfer.generate(
    "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    "Explain AI in one sentence.",
    max_tokens=50,
)
print(response)

# Generate on CPU
print("\n=== CPU Generation ===")
engine_cpu = uniinfer.Engine(
    model="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    device="cpu",
)
result = engine_cpu.generate("Explain AI in one sentence.", max_tokens=50)
print(result.text)
engine_cpu.close()

# Streaming example
print("\n=== Streaming (GPU) ===")
engine_gpu = uniinfer.Engine(
    model="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    device="auto",
)
for chunk in engine_gpu.stream("Write a short poem about hardware.", max_tokens=100):
    print(chunk.text, end="", flush=True)
print()
engine_gpu.close()
```

Run it:

```bash
python test_uniinfer.py
```

---

## Step 7: List and Manage Cached Models

### List what you have downloaded

```bash
uniinfer list
```

### Models are cached at

```
C:\Users\Iulian\.uniinfer\cache\models\
```

You can delete model folders manually to free disk space.

---

## What to Look For

### GPU test (CUDA)
- Loading should mention GPU layers being offloaded
- Generation should be fast (50+ tokens/sec for TinyLlama on RTX 3060)
- VRAM usage will increase while model is loaded

### CPU test
- Same model, same output quality
- Slower speed (5-15 tokens/sec for TinyLlama on your AMD Ryzen)
- Uses system RAM instead of VRAM

### The key point
The exact same command works on both devices. Change `--device cuda:0` to `--device cpu` and everything still works. No code changes, no format conversion, no different libraries. That is the abstraction layer in action.

---

## Troubleshooting

### "uniinfer" is not recognized
Use the full path:
```
C:\Users\Iulian\AppData\Roaming\Python\Python312\Scripts\uniinfer.exe
```
Or add the Scripts folder to PATH (see Setup section above).

### Model download fails
Check your internet connection. HuggingFace may require authentication for some models:
```bash
pip install huggingface-hub
huggingface-cli login
```

### Out of memory on GPU
Use a smaller quantization:
```bash
uniinfer generate --model "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF" --prompt "Hello" --quantization q4_k_m
```

### Slow on CPU
This is expected. CPU inference is 5-10x slower than GPU. Use smaller models (TinyLlama) for CPU testing.
