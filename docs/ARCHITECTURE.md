# UniInfer: Hardware-Agnostic AI Inference Runtime — Technical Architecture

## 1. System Architecture

```
                            ┌─────────────────────────────────────────────┐
                            │            Developer Interface              │
                            │                                             │
                            │   Python SDK    REST API    gRPC Endpoint   │
                            └──────────┬──────────┬──────────┬────────────┘
                                       │          │          │
                                       ▼          ▼          ▼
                            ┌─────────────────────────────────────────────┐
                            │             API Gateway / Router            │
                            │                                             │
                            │  - Request validation & auth                │
                            │  - Model routing (which model, which node)  │
                            │  - Load balancing across devices            │
                            └─────────────────────┬───────────────────────┘
                                                  │
                                                  ▼
                            ┌─────────────────────────────────────────────┐
                            │           Inference Orchestrator            │
                            │                                             │
                            │  - Continuous batching scheduler            │
                            │  - KV cache manager                         │
                            │  - Token streaming coordinator              │
                            │  - Request lifecycle (queuing → execution)  │
                            └─────────────────────┬───────────────────────┘
                                                  │
                           ┌──────────────────────┼──────────────────────┐
                           │                      │                      │
                           ▼                      ▼                      ▼
               ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐
               │  Model Registry   │  │  Execution Engine  │  │  Memory Manager   │
               │                   │  │                    │  │                   │
               │ - Model catalog   │  │ - Graph executor   │  │ - Device memory   │
               │ - Format convert  │  │ - Op dispatch      │  │   allocation      │
               │ - Weight cache    │  │ - Kernel selection  │  │ - KV cache pool   │
               │ - Quantization    │  │ - Fusion passes    │  │ - Paged attention  │
               │ - GGUF/ONNX/ST    │  │                    │  │ - Swap to host    │
               └───────┬───────────┘  └────────┬───────────┘  └────────┬──────────┘
                       │                       │                       │
                       │                       ▼                       │
                       │          ┌─────────────────────────┐          │
                       │          │ Hardware Abstraction     │          │
                       │          │ Layer (HAL)              │          │
                       │          │                          │          │
                       │          │ Unified interface for:   │◄─────────┘
                       │          │  - Tensor ops            │
                       │          │  - Memory alloc/free     │
                       │          │  - Stream/queue mgmt     │
                       │          │  - Device enumeration    │
                       │          └──────────┬──────────────┘
                       │                     │
                       │      ┌──────────────┼──────────────┬──────────────┐
                       │      │              │              │              │
                       │      ▼              ▼              ▼              ▼
                       │ ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
                       │ │  CUDA   │   │  ROCm   │   │ Vulkan  │   │  CPU    │
                       │ │ Adapter │   │ Adapter │   │ Adapter │   │ Adapter │
                       │ │         │   │         │   │         │   │         │
                       │ │cuBLAS   │   │rocBLAS  │   │Kompute/ │   │OpenBLAS │
                       │ │cuDNN    │   │MIOpen   │   │vulkan.h │   │oneDNN   │
                       │ │TensorRT │   │         │   │         │   │AMX/AVX  │
                       │ └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘
                       │      │             │             │             │
                       │      ▼             ▼             ▼             ▼
                       │ ┌────────────────────────────────────────────────┐
                       └►│              Physical Hardware                 │
                         │  NVIDIA GPU │ AMD GPU │ Intel GPU │ CPU        │
                         └────────────────────────────────────────────────┘
```

## 2. Component Breakdown

### 2.1 API Gateway / Router
- **What**: Accepts inference requests (REST, gRPC, Python in-process), validates inputs, routes to correct model/device
- **Build or Wrap**: BUILD — thin layer on FastAPI (REST) + grpcio (gRPC)
- **Endpoints**: `POST /v1/completions`, `POST /v1/chat/completions` (OpenAI-compatible), Python `UniInfer.generate()`

### 2.2 Inference Orchestrator
- **What**: Central brain. Manages request lifecycle: queuing, batch formation, decode loop, KV cache allocation, token streaming
- **Build or Wrap**: BUILD — inspired by vLLM's scheduler design (~2000 lines of Python), reimplemented for hardware-agnostic memory management
- **Key interfaces**: `Scheduler.add_request()`, `Scheduler.step()`, `Scheduler.abort()`

### 2.3 Model Registry
- **What**: Downloads models from HuggingFace, converts to internal format (GGUF/ONNX), manages weight caching, applies quantization
- **Build or Wrap**: WRAP HuggingFace `huggingface_hub` for downloads + BUILD conversion pipeline
- **Key interfaces**: `ModelRegistry.load(model_id, backend, quantization)`

### 2.4 Execution Engine
- **What**: Takes loaded model + batch of token sequences, runs one forward pass, returns logits
- **Build or Wrap**: WRAP — delegates to llama.cpp (via llama-cpp-python) and ONNX Runtime. We do NOT write our own compute kernels
- **Key interfaces**: `ExecutionBackend.load_model()`, `ExecutionBackend.forward()`

### 2.5 Memory Manager
- **What**: Allocates and tracks GPU/CPU memory. Manages KV cache as block pool. Handles swap to host RAM under memory pressure
- **Build or Wrap**: BUILD — must be hardware-aware with unified interface

### 2.6 Hardware Abstraction Layer (HAL)
See Section 3.

---

## 3. Hardware Abstraction Layer

### 3.1 DeviceAdapter Interface

```
DeviceAdapter (abstract):
    # Discovery
    get_device_count() -> int
    get_device_info(device_id) -> DeviceInfo

    # Memory management
    allocate(size_bytes, device_id) -> MemoryHandle
    free(handle)
    copy_host_to_device(host_ptr, device_handle, size)
    copy_device_to_host(device_handle, host_ptr, size)
    get_free_memory(device_id) -> int

    # Execution
    create_stream() -> StreamHandle
    synchronize(stream)
    destroy_stream(stream)

    # Backend runtime
    get_supported_backends() -> List[BackendType]
    create_execution_backend(backend_type) -> ExecutionBackend
```

### 3.2 Adapter Implementations

| Adapter | Discovery | Compute Libraries | Backends Supported |
|---------|-----------|-------------------|-------------------|
| **CUDAAdapter** | pynvml | cuBLAS, cuDNN, TensorRT | llama.cpp (CUDA), ORT (TensorRT EP) |
| **ROCmAdapter** | rocm-smi | rocBLAS, MIOpen | llama.cpp (ROCm/HIP), ORT (ROCm EP) |
| **VulkanAdapter** | vulkaninfo | Kompute | llama.cpp (Vulkan) |
| **CPUAdapter** | psutil + platform | OpenBLAS, oneDNN | llama.cpp (CPU), ORT (CPU/OpenVINO EP) |

### 3.3 Hardware Discovery & Selection

1. Probe in order: CUDA → ROCm → Vulkan → CPU
2. Query capabilities (memory, compute power, supported precisions)
3. Build DeviceMap ranked by estimated throughput
4. Default: pick fastest GPU, fall back to CPU
5. User override: `UniInfer(device="rocm:0")` or `device="cpu"`

---

## 4. Model Loading Pipeline

```
HuggingFace Model (safetensors)
         │
    Format Detector (architecture, dtype, layer count)
         │
         ├── target = llama.cpp → Convert to GGUF (llama.cpp's convert_hf_to_gguf.py)
         ├── target = ONNX Runtime → Convert to ONNX (optimum library)
         └── target = native PyTorch → Keep safetensors (CUDA fallback only)
```

### Quantization Strategy

| Precision | GGUF Format | When to Use |
|-----------|------------|-------------|
| FP16 | f16 | GPUs with >=24GB VRAM |
| INT8 | Q8_0 | GPUs with 8-16GB VRAM |
| INT4 | Q4_K_M | Consumer GPUs, 4-8GB VRAM |

Auto-selection: if FP16 model exceeds 80% of device memory, step down to INT8, then INT4.

### Cache Structure

```
~/.uniinfer/cache/
  └── models/
      └── meta-llama--Llama-3.1-8B-Instruct/
          ├── gguf/q4_k_m.gguf
          ├── onnx/model_fp16.onnx
          └── metadata.json
```

---

## 5. LLM-Specific Components

### KV Cache Management
- **MVP (llama.cpp backend)**: Use llama.cpp's built-in contiguous KV cache. Works on all backends.
- **v1.0 (CUDA only)**: Paged attention via FlashInfer for higher concurrency.

### Continuous Batching
Three queues: Waiting → Running → Preempted. Each scheduler step:
1. Admit waiting requests if KV cache blocks available
2. Preempt lowest-priority if memory tight
3. Form batch from all running requests
4. Execute one forward pass
5. Sample tokens, check stop conditions
6. Stream completed tokens

### Token Streaming
- REST: Server-Sent Events (OpenAI streaming format)
- gRPC: Server streaming RPC
- Python: `async for token in engine.astream(prompt)`

---

## 6. API Design

### Python SDK

```python
import uniinfer

# One-liner
response = uniinfer.generate("meta-llama/Llama-3.1-8B-Instruct", "What is gravity?")

# Engine for server usage
engine = uniinfer.Engine(
    model="meta-llama/Llama-3.1-8B-Instruct",
    device="auto",
    quantization="auto",
)

# Generate
output = engine.generate(prompt="Explain quantum computing.", max_tokens=512)

# Stream
for chunk in engine.stream("Write a haiku about code."):
    print(chunk.text, end="", flush=True)

# Chat (OpenAI-compatible)
messages = [{"role": "user", "content": "Hello"}]
response = engine.chat(messages, max_tokens=100)

# Device inspection
print(uniinfer.devices())

# Explicit device
engine = uniinfer.Engine("mistralai/Mistral-7B-v0.1", device="vulkan:0")
```

### REST API (OpenAI-Compatible)

```bash
uniinfer serve --model meta-llama/Llama-3.1-8B-Instruct --port 8000
```

### CLI

```bash
uniinfer chat --model meta-llama/Llama-3.1-8B-Instruct
uniinfer devices
uniinfer pull meta-llama/Llama-3.1-8B-Instruct --quantization q4_k_m
uniinfer bench --model meta-llama/Llama-3.1-8B-Instruct --device auto
```

---

## 7. Performance Strategy

| Source of Overhead | Impact | Mitigation |
|-------------------|--------|------------|
| Abstraction dispatch | <0.1% | One virtual call per forward pass |
| Format conversion | 0% runtime | Offline, cached |
| llama.cpp vs hand-tuned CUDA | 5-15% | llama.cpp uses cuBLAS + FlashAttention |
| Contiguous vs paged KV cache | 5-20% at high concurrency | Accept for MVP, paged attention in v1.0 |
| Python scheduling | 1-3% | GPU compute >> Python overhead |

**Core rule**: We write ZERO compute kernels. We orchestrate. All heavy computation runs in C/C++ libraries that release the GIL.

---

## 8. Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| HTTP server | FastAPI + uvicorn |
| Async runtime | asyncio + uvloop |
| Configuration | pydantic v2 |
| Key dependencies | llama-cpp-python, onnxruntime, huggingface-hub, tokenizers |
| Build | uv + hatchling |
| CI/CD | GitHub Actions (matrix: os × gpu backend) |
| Linting | ruff |
| Type checking | mypy (strict) |
| Testing | pytest |

---

## 9. Project Structure

```
uniinfer/
├── pyproject.toml
├── Dockerfile
├── src/uniinfer/
│   ├── __init__.py              # Public API
│   ├── api/
│   │   ├── server.py            # FastAPI app
│   │   ├── routes_completions.py
│   │   ├── routes_models.py
│   │   ├── schemas.py           # Pydantic request/response models
│   │   └── streaming.py         # SSE helpers
│   ├── engine/
│   │   ├── engine.py            # Main Engine class
│   │   ├── scheduler.py         # Continuous batching
│   │   ├── sampling.py          # Temperature, top-p, top-k
│   │   └── stop_criteria.py
│   ├── models/
│   │   ├── registry.py          # Download, cache, catalog
│   │   ├── converter.py         # Format conversion dispatch
│   │   ├── converter_gguf.py
│   │   ├── converter_onnx.py
│   │   └── quantization.py
│   ├── hal/
│   │   ├── interface.py         # DeviceAdapter ABC
│   │   ├── discovery.py         # Hardware probing
│   │   ├── selector.py          # Device selection
│   │   ├── cuda_adapter.py
│   │   ├── rocm_adapter.py
│   │   ├── vulkan_adapter.py
│   │   └── cpu_adapter.py
│   ├── backends/
│   │   ├── interface.py         # ExecutionBackend ABC
│   │   ├── llamacpp.py
│   │   ├── onnxrt.py
│   │   └── benchmark.py
│   ├── memory/
│   │   ├── manager.py
│   │   ├── kv_cache.py
│   │   └── pool.py
│   ├── config/
│   │   ├── engine_config.py
│   │   └── serving_config.py
│   ├── metrics/
│   │   └── prometheus.py
│   └── cli/
│       └── main.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── benchmarks/
└── docs/
    ├── ARCHITECTURE.md
    └── API_REFERENCE.md
```

---

## 10. MVP Scope

### v0.1 — Proof of Concept

**Goal**: Run any LLM on any hardware with one line of Python.

**Includes**:
- `uniinfer.Engine(model, device="auto")` with auto-detection
- Single backend: llama.cpp (covers CUDA, ROCm, Vulkan, CPU)
- `engine.generate()` and `engine.stream()`
- Auto-download from HuggingFace, auto-convert to GGUF, disk cache
- Auto-quantization based on available memory
- Hardware discovery and CLI (`uniinfer chat`, `uniinfer devices`)

**Excludes**: REST API, continuous batching, ONNX backend, gRPC, paged attention, multi-GPU

**Success criteria**: `pip install uniinfer` then one line of code works on NVIDIA, AMD, and CPU. Performance within 5% of raw llama.cpp.

### v0.5 — Usable Server

**Adds**: OpenAI-compatible REST API, continuous batching, concurrent users, ONNX Runtime backend, Prometheus metrics, Docker images

**Success criteria**: 10+ concurrent users, throughput within 20% of vLLM on NVIDIA, works on AMD where vLLM doesn't

### v1.0 — Production Ready

**Adds**: gRPC, paged attention (CUDA), multi-GPU tensor parallelism, speculative decoding, LoRA hot-loading, structured output, Kubernetes Helm chart

**Success criteria**: Drop-in vLLM replacement with broader hardware support. Within 10% of vLLM on NVIDIA, functional on AMD/Intel.

---

## Hard Problems (Honest Assessment)

1. **Paged attention on non-CUDA is unsolved** — no production-quality ROCm/Vulkan implementation exists
2. **llama.cpp is our biggest dependency AND risk** — fast-moving, frequent breaking changes
3. **Model architecture coverage** depends on llama.cpp upstream support
4. **Vulkan is genuinely 20-40% slower** than native CUDA — inherent limitation, must be transparent
5. **Python GIL is NOT a bottleneck** — all heavy compute happens in C/C++ libraries
