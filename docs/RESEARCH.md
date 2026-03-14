# Research Phase: AI Compute Abstraction Landscape

## Existing Projects Analysis

| Project | Layer | What It Does | Hardware Breadth | Training | Inference | Maturity |
|---------|-------|-------------|-----------------|----------|-----------|----------|
| **OpenXLA/XLA** | Compiler | Google's ML compiler, TPU-first | Medium | Yes | Yes | Production (TPU/NVIDIA) |
| **MLIR** | Infrastructure | Framework for building compilers | N/A (enables others) | N/A | N/A | Production |
| **Triton** | Kernel compiler | Python GPU kernel language | Low (NVIDIA, AMD, Intel early) | Yes | Yes | Production (NVIDIA) |
| **TVM** | End-to-end compiler | Auto-tuning for edge/mobile | High | Limited | Yes | Production (edge) |
| **ONNX Runtime** | Inference runtime | Cross-platform model deployment | High | Minimal | Yes | Production |
| **ROCm** | Full platform | AMD's CUDA equivalent | AMD only | Yes | Yes | Production (datacenter) |
| **oneAPI/SYCL** | Programming model | Intel cross-architecture API | Medium (Intel-focused) | Limited | Yes | Mixed |
| **Vulkan/WebGPU** | Low-level API | Universal GPU compute access | Very High | No | Yes (limited) | Vulkan mature, WebGPU early |

## The 5 Critical Gaps (Our Opportunity)

### Gap 1: No Portable Communication Layer
NCCL (multi-GPU communication) is NVIDIA-only. AMD's RCCL trails. No neutral standard.

### Gap 2: No Training Abstraction
Inference portability is ~80% solved. Training portability is ~10% solved.

### Gap 3: No Hardware Topology Discovery
Engineers manually configure parallelism strategies. No auto-discovery and optimization.

### Gap 4: No Cross-Hardware Validation
No standard way to verify training produces same results across hardware vendors.

### Gap 5: No Vendor-Neutral Governance
Every player (Google, NVIDIA, AMD, Intel) wants their own lock-in. Nobody built the neutral layer.

## Why Existing Solutions Haven't Won

1. **Every hardware vendor wants their OWN lock-in** — not neutrality
2. **Performance fear** — any abstraction adds overhead, at scale even 5% = millions of dollars
3. **Ecosystem inertia** — millions of CUDA developers, thousands of libraries
4. **Each project solves only one slice** — nobody spans the full stack
5. **The "it works" problem** — switching introduces risk with speculative benefits

## Why Now Is The Right Time

- GPU costs are extreme ($25-40k per H100) and supply is constrained
- AMD hardware (MI300X) is finally competitive
- llama.cpp proved multi-backend inference is viable
- Economic pressure is mounting for hardware alternatives
- The open-source community is increasingly interested in hardware independence

## Strategic Decision

Focus on **inference first** (not training):
- Simpler problem (no autograd, no gradient sync)
- Companies spend money on inference every day (ongoing cost)
- Building blocks exist (llama.cpp, ONNX Runtime)
- Clear monetization: reduce enterprise compute costs
