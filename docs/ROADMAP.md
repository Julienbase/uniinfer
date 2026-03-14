# UniInfer Roadmap

## Strategic Direction

UniInfer's lane: **"Any model, any hardware, zero config."**

We don't compete with vLLM/TGI on multi-GPU enterprise scale. We compete on **intelligence** — knowing what fits your hardware, auto-configuring, and giving actionable guidance instead of cryptic errors or silent OOM crashes.

**One-liner**: *"Ollama makes it easy to run models. UniInfer makes it impossible to run them wrong."*

---

## Completed Milestones

### v0.1 — Proof of Concept ✅
- Hardware Abstraction Layer (CUDA, ROCm, Vulkan, CPU detection)
- llama-cpp-python backend with auto GPU offloading
- Auto-download from HuggingFace, auto-convert to GGUF, disk cache
- Auto-quantization based on available memory
- Python SDK: `Engine.generate()`, `Engine.stream()`, `Engine.chat()`
- CLI: `uniinfer chat`, `uniinfer devices`, `uniinfer pull`, `uniinfer bench`

### v0.5 — Usable Server ✅
- OpenAI-compatible REST API (FastAPI + uvicorn)
- Async request scheduler with backpressure (concurrent users queued)
- SSE streaming (`data: {json}\n\n` format)
- ONNX Runtime backend + backend registry
- Prometheus metrics (requests, tokens, latency, queue depth)
- `uniinfer serve` CLI command
- Docker images (CPU + NVIDIA GPU variants)
- Full test suite (unit + integration)

---

## v1.0 — Smart Runtime ("It Just Works") ✅

### The Problem We Solved

Real user pain points that nobody solved well:

1. **"Which model fits my hardware?"** — Users download 37GB files that won't load. LM Studio shows "likely too large" but doesn't suggest what WILL fit.
2. **"I switched GPU / reinstalled drivers and everything broke"** — CUDA mismatches, ROCm kernel issues, hours of debugging.
3. **"I want to use a local model in my app but integration is painful"** — Start a server, manage its lifecycle, handle connections...
4. **"I found a model but what format is it?"** — GGUF, GPTQ, AWQ, EXL2, SafeTensors, ONNX — users don't know which to pick.
5. **"My model works but it's slow and I don't know why"** — No clear diagnostics on GPU utilization, layer offloading, or bottlenecks.

### Feature 1: Smart Model Fitting ✅

**VRAM budget calculator** with pre-download validation — never waste bandwidth on models that won't fit.

**What was built**:
- `models/fitting.py` — `check_model_fit()` calculates `model_size + kv_cache + overhead <= available_memory × safety_factor`
- `models/gguf_metadata.py` — Binary GGUF header parser (architecture, quantization, context length, tensor count)
- `QUANT_BYTES_PER_PARAM` table for 12 quantization levels (f32 through q2_k)
- `FitReport` with alternatives table showing which quantizations fit
- Pre-download validation in `download_model()` — raises `ModelTooLargeError` before downloading
- `uniinfer run --dry-run` CLI command with Rich panel showing fit report + quantization options
- Integrated into Engine setup: pre-load estimate + post-download actual size check

### Feature 2: Multi-Format Loading ✅

**Drop in any format** — UniInfer auto-detects and routes to the right backend.

**What was built**:
- `backends/transformers_backend.py` — Full `ExecutionBackend` wrapping HuggingFace transformers (`AutoModelForCausalLM`, `AutoTokenizer`)
- Magic byte detection (`detect_backend_from_magic()`) for format-agnostic loading
- Directory detection (`_has_safetensors()`) for HuggingFace model directories
- Backend registry updated: `.gguf` → llamacpp, `.onnx` → onnxruntime, directories/SafeTensors → transformers
- `trust_remote_code=False` enforced for security

### Feature 3: Hardware Resilience ✅

**Automatic fallback chain** — if your preferred device fails, UniInfer tries the next one.

**What was built**:
- `hal/health.py` — Device health checks (CUDA via pynvml, ROCm via rocm-smi, Vulkan via vulkaninfo, CPU via psutil)
- `engine/fallback.py` — `try_with_fallback()` builds ordered chain (CUDA → ROCm → Vulkan → CPU), skips unhealthy devices, retries on alternatives
- `FallbackResult` tracks events, health reports, and provides summary
- Engine `_load_model_with_fallback()` wraps model loading with automatic retry
- Fallback events exposed via `engine.info()["fallback"]`

### Feature 4: Inference Diagnostics ✅

**Per-call timing and session metrics** — know exactly how fast your model runs.

**What was built**:
- `engine/diagnostics.py` — `InferenceMetrics` (per-call: elapsed, tok/s, method) + `SessionDiagnostics` (aggregate: total tokens, avg/peak tok/s, load time)
- All four inference methods (`generate`, `stream`, `chat`, `chat_stream`) auto-instrumented with `time.perf_counter()`
- Model load time tracked
- Full diagnostics exposed via `engine.info()["diagnostics"]` dict

### Feature 5: One-Line SDK Enhancement ✅

**Simplified aliases and top-level functions** — minimum cognitive load.

**What was built**:
- `models/aliases.py` — 7 model aliases (tinyllama-1b through llama-3.3-70b) with param counts, default quant, context length
- `uniinfer.chat(model, message)` — one-line chat, creates temporary Engine, returns string
- `uniinfer.chat_stream(model, message)` — streaming variant
- `uniinfer aliases` CLI command with Rich table
- Alias resolution integrated into Engine constructor

---

## v1.5 — Management Layer (Planned)

- Web dashboard (browser-based, served by UniInfer)
- Model management UI (download, delete, compare)
- Usage analytics (requests/day, tokens generated, cost estimates)
- Multi-model serving with auto-routing
- Model performance comparison tools

**Purpose**: Transition from CLI tool to visual product. This is where teams start adopting UniInfer.

---

## v2.0 — Enterprise (Open Core)

- RBAC + API key management (team access control)
- Audit logs (who ran what model, when, with what data)
- SSO integration (SAML, OIDC)
- Air-gapped deployment mode (no network calls, pre-bundled models)
- Priority support + SLA contracts

**Target**: Companies running local AI for privacy/compliance — healthcare, legal, finance, government.

**Pricing**: $500-2000/month per deployment.

---

## What We Deliberately Don't Build

| Feature | Why Not | Who Does It Better |
|---------|---------|-------------------|
| Multi-GPU tensor parallelism | llama.cpp doesn't support it; adding vLLM as a wrapper makes UniInfer a thin shell | vLLM, TGI |
| Training | Different problem, different market, massive scope | PyTorch, DeepSpeed |
| Model fine-tuning | Inference runtime, not a training platform | Axolotl, Unsloth |
| Kubernetes orchestration | Infrastructure layer, not our lane | K8s, Helm, ArgoCD |
| Custom CUDA kernels | We orchestrate, we don't compute | llama.cpp, FlashAttention |

**Philosophy**: Do one thing exceptionally well. Be the smartest single-GPU inference runtime, not a mediocre everything platform.

---

## Multi-GPU Decision Record

**Decision**: Stay single-GPU. Do not add multi-GPU tensor parallelism.

**Research findings** (March 2026):
- llama-cpp-python does NOT support true tensor parallelism (8-10x slowdown with `tensor_split`)
- vLLM is the proven multi-GPU solution but adding it makes UniInfer a wrapper, not a product
- TensorRT-LLM is NVIDIA-only (conflicts with hardware-agnostic mission)
- ExLlamaV2 doesn't support GGUF with tensor parallelism

**Rationale**: UniInfer's value is "any model, any hardware, zero config" — that's the single-GPU sweet spot. Enterprises needing 8xH100 tensor parallelism should use vLLM directly. Trying to compete there with a solo dev against 400+ contributors is not viable.

**Revisit when**: llama.cpp merges proper tensor parallelism (ik_llama.cpp fork), or if user demand proves this wrong.
