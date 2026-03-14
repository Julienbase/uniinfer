# UniInfer: Who Benefits and Market Positioning

## Three Customer Segments

### 1. Companies Running AI Infrastructure (Primary Target)

Companies spending $100K–$50M+/year on GPU compute for AI inference.

| Who | Pain Point | How UniInfer Helps |
|-----|-----------|-------------------|
| SaaS companies serving AI features (chatbots, copilots, search) | Locked into NVIDIA, can't negotiate prices, supply-constrained | Run on AMD/Intel hardware at 30-50% lower cost |
| AI startups (serving LLMs via API) | GPU costs eat 60-80% of revenue | Hardware flexibility = cost optimization |
| Enterprises deploying internal AI | IT policy may prefer vendor diversity, or they already own AMD/Intel hardware | Use existing hardware instead of buying NVIDIA |
| Managed service providers / cloud resellers | Want to offer AI compute without NVIDIA dependency | White-label the runtime |

**Real example**: A company running Llama 3 on 50 NVIDIA A100s at ~$2/hr each = **$876K/year**. If they move half to AMD MI300X instances (30% cheaper), they save **$130K/year**. UniInfer charges a fraction of that savings.

### 2. Developers and AI Engineers (Adoption Driver)

These are not the buyers — they are the users who bring the tool into organizations.

| Who | Pain Point | How UniInfer Helps |
|-----|-----------|-------------------|
| ML engineers deploying models | Must write different code for CUDA vs ROCm vs CPU | One API, any hardware |
| Solo developers / indie hackers | Can't afford NVIDIA GPUs, have AMD or Intel at home | Their hardware just works |
| Researchers at universities | Limited GPU budgets, diverse hardware in labs | Run experiments on whatever is available |

Developers adopt the open-source tool for free, build on it, and bring it into their companies. This is the Red Hat / Docker adoption playbook.

### 3. Hardware Vendors (Strategic Allies)

These are potential partners and investors, not direct customers.

| Who | Pain Point | How UniInfer Helps |
|-----|-----------|-------------------|
| AMD | Sells competitive hardware but CUDA lock-in blocks adoption | Makes AMD GPUs a first-class AI target |
| Intel | Same problem, worse market position | Gives Intel GPUs a reason to exist for AI workloads |
| Cloud providers (Oracle, CoreWeave, Lambda) | Want to differentiate with non-NVIDIA GPU options | Makes their AMD/Intel instances usable for AI out of the box |

AMD has a direct financial interest in tools that make their GPUs work seamlessly for AI. This represents a funding and partnership angle.

---

## Who Does NOT Benefit

| Who | Why Not |
|-----|---------|
| NVIDIA | UniInfer reduces their lock-in — they will not support this |
| Google / Amazon / Microsoft | Building vertical stacks (TPU / Trainium / Maia) — they want their own lock-in, not neutrality |
| Consumers using ChatGPT or similar APIs | They interact through APIs and never see hardware — irrelevant to them |
| Companies focused on AI training (initially) | UniInfer v1 targets inference only — training abstraction comes later |

---

## Go-To-Market Sequence

```
Phase 1: Developers adopt open-source tool (free)
         ↓
Phase 2: Developers bring it into their companies
         ↓
Phase 3: Companies need support, SLAs, dashboards → pay for enterprise tier
         ↓
Phase 4: Hardware vendors partner and sponsor for ecosystem growth
```

This follows the proven path of Docker, Kubernetes, Redis, and Terraform.

---

## The Pitch

**For companies**: "Run your AI models on the cheapest hardware available. Stop paying the NVIDIA tax."

**For developers**: "One line of Python. Any model. Any hardware. Zero configuration."

**For hardware vendors**: "We make your chips a viable AI platform overnight."

---

## Monetization Models

| Model | Precedent | Revenue Type |
|-------|-----------|-------------|
| Open-core with enterprise features (support, SLAs, admin dashboard, SSO) | Red Hat (sold for $34B) | Subscriptions |
| Managed runtime service | Databricks (built on open-source Spark) | Usage-based pricing |
| Hardware-agnostic compute broker (route workloads to cheapest available hardware) | Cloudflare Workers model | Margin on compute |

---

## Key Economics

- NVIDIA H100: $25,000–$40,000 per unit
- NVIDIA cloud instances: $2–$4/hr per GPU
- AMD MI300X cloud instances: 20-40% cheaper than equivalent NVIDIA
- Intel and CPU instances: significantly cheaper but lower throughput

The cost gap between NVIDIA and alternatives is the economic engine that drives adoption. As long as NVIDIA GPUs remain expensive and supply-constrained, demand for hardware-agnostic inference grows.
