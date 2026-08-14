# SHMQ-Ultimate

**Static Hierarchical Mix-precision Quantization for LLMs — Improved Reimplementation**

Combines the best of 5 sources into a unified W4.8A8 quantization pipeline:

| Source | Component Used |
|--------|----------------|
| [HAWQ-V3](https://github.com/Zhen-Dong/HAWQ) | ILP bit allocation (PULP), PyHessian trace (alternative) |
| [SliM-LLM](https://github.com/Aaronhuang-778/SliM-LLM) | AutoGPTQ + per-element OBS Hessian (Eq. 10) + SQC calibration |
| [AutoRound](https://github.com/intel/auto-round) | SignSGD learnable rounding (200 steps), V baking |
| [SmoothQuant](https://github.com/mit-han-lab/smoothquant) | Activation outlier migration (pre-processing) |
| [SHMQ paper](https://aclanthology.org/2025.emnlp-industry.175/) | Decoupled permutation + PermutedRMSNorm + parallel constraint (custom) |

**Reference paper:** Yi Zhang et al., "Beyond Dynamic Quantization: An Efficient Static Hierarchical Mix-precision Framework for Near-Lossless LLM Compression", EMNLP Industry 2025.

---

## Pipeline (9 steps)

```
                 ┌─────────────────────────────────────────┐
   FP16 LLM ───► │ Step 1: SmoothQuant pre-processing       │
                 │         (migrate activation outliers)    │
                 └────────────────┬─────────────────────────┘
                                  ▼
                 ┌─────────────────────────────────────────┐
                 │ Step 2: Sensitivity computation          │
                 │   - Inter-layer: Fisher (Eq. 7)          │
                 │   - Intra-layer: OBS Hessian (Eq. 10)    │
                 │   - Manhattan aggregation (Eq. 11)       │
                 │   - Parallel constraint (avg + concat)   │
                 └────────────────┬─────────────────────────┘
                                  ▼
                 ┌─────────────────────────────────────────┐
                 │ Step 3: ILP bit allocation {4, 8}        │
                 │   (HAWQ-V3 PULP solver, parallel eq.)    │
                 └────────────────┬─────────────────────────┘
                                  ▼
                 ┌─────────────────────────────────────────┐
                 │ Step 4: Decoupled permutation            │
                 │   (sort by sens → partition → sort by    │
                 │    magnitude) — SHMQ Eq. 12              │
                 └────────────────┬─────────────────────────┘
                                  ▼
                 ┌─────────────────────────────────────────┐
                 │ Step 5: Permutation fusion into RMSNorm  │
                 │   (zero-overhead at inference)           │
                 └────────────────┬─────────────────────────┘
                                  ▼
                 ┌─────────────────────────────────────────┐
                 │ Step 6: AutoRound SignSGD (200 steps)    │
                 │   (learnable V, baked into weights)      │
                 └────────────────┬─────────────────────────┘
                                  ▼
                 ┌─────────────────────────────────────────┐
                 │ Step 7: SQC calibration                  │
                 │   (salience-weighted scale optimization) │
                 └────────────────┬─────────────────────────┘
                                  ▼
                 ┌─────────────────────────────────────────┐
                 │ Step 8: Mixed INT4/INT8 quantization     │
                 │   (GPTQ for 4-bit, RTN for 8-bit)        │
                 └────────────────┬─────────────────────────┘
                                  ▼
                          Quantized LLM
                       (W4.8A8, near-lossless)
```

---

## Hyperparameters (from SHMQ paper)

| Parameter | Value | Reference |
|-----------|-------|-----------|
| Format | W4.8A8 (W4A8 + 20% W8A8) | §4.1 |
| `target_hp_ratio` (Ut) | 0.20 | §4.1 |
| `base_hp_ratio` (Ub) | 0.125 | App. A.3.1 |
| `dampening` (λ) | 0.1 | Eq. 10 |
| Group size | 128 | §4.1 |
| Calibration | 128 samples × 2048 tokens WikiText-2 | §4.1 |
| AutoRound iters | 200 | AutoRound paper |
| SmoothQuant α | 0.5 (default; 0.6-0.9 for W8A8) | SmoothQuant paper |
| Inter-layer Hessian | Fisher (Eq. 7) | App. A.2 |
| Intra-layer Hessian | H = X X^T (Eq. 10) | App. A.2 |

---

## Quick Start

### Installation

```bash
cd /home/z/my-project/shmq-ultimate
pip install torch transformers datasets pulp pymupdf pyhessian
```

### Run on Qwen2.5-7B-Instruct (requires GPU)

```python
from shmq import SHMQConfig, SHMQPipeline

config = SHMQConfig(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    device="cuda",
    dtype="float16",
    n_samples=128,
    sequence_length=2048,
    autoround_iters=200,
    target_hp_ratio=0.20,
)

pipeline = SHMQPipeline(config)
pipeline.run()
pipeline.save_model("./qwen2.5-7b-shmq-ultimate")
```

### Run smoke tests (CPU)

```bash
cd /home/z/my-project/shmq-ultimate
python3 tests/test_smoke.py        # 15 module tests, ~10 seconds
python3 tests/test_e2e_quick.py    # E2E on Qwen2.5-0.5B (2 blocks), ~90 seconds
```

---

## Project Structure

```
shmq-ultimate/
├── src/shmq/                       # Main package
│   ├── __init__.py                 # Exports SHMQConfig, SHMQPipeline
│   ├── config.py                   # SHMQConfig dataclass (all hyperparameters)
│   ├── model_loader.py             # ModelLoader — loads HF LLM, identifies layers
│   ├── calibration.py              # WikiText-2 / C4 / Pile calibration data
│   ├── utils.py                    # Shared utilities (quantize, topk, etc.)
│   ├── pipeline.py                 # SHMQPipeline — orchestrates 9 steps
│   │
│   ├── smooth/                     # Step 1: SmoothQuant (vendored from MIT-Han-Lab)
│   │   ├── smooth.py               # smooth_ln_fcs_llama_like, smooth_lm
│   │   └── calibration.py          # get_act_scales (activation scale hook)
│   │
│   ├── sensitivity/                # Step 2: Sensitivity computation
│   │   ├── fisher.py               # Inter-layer Fisher (Eq. 7)
│   │   ├── pyhessian_trace.py      # PyHessian alternative (HAWQ-V3 style)
│   │   ├── obs.py                  # Intra-layer OBS Hessian (Eq. 10) + Cholesky inverse
│   │   ├── manhattan.py            # Manhattan norm aggregation (Eq. 11)
│   │   └── parallel.py             # Parallel constraint (avg + concat)
│   │
│   ├── ilp/                        # Step 3: ILP bit allocation
│   │   └── solver.py               # PULP solver, 2 levels {4,8}, parallel eq.
│   │
│   ├── permutation/                # Steps 4-5: Decoupled permutation + fusion
│   │   ├── metric.py               # Permutation metric: act × weight l∞
│   │   ├── decoupled.py            # Decoupled permutation algorithm (SHMQ §3.2.3)
│   │   └── rmsnorm_fusion.py       # PermutedRMSNorm — zero-overhead fusion
│   │
│   ├── autoround/                  # Step 6: AutoRound SignSGD
│   │   ├── sign_sgd.py             # SignSGD optimizer (θ ← θ - lr·sign(g))
│   │   ├── wrapper.py              # WrapperLinear with learnable V
│   │   ├── baking.py               # Bake V into weights (zero overhead)
│   │   └── autoround_block.py      # Per-block 200-step optimization
│   │
│   └── quantize/                   # Steps 7-8: SQC + GPTQ + mixed quant
│       ├── sqc.py                  # Salience-Weighted Quantizer Calibration
│       ├── gptq.py                 # GPTQ OBS weight update (Frantar 2023)
│       └── mixed.py                # Mixed INT4/INT8 final quantization
│
├── tests/
│   ├── test_smoke.py               # 15 module-level tests
│   └── test_e2e_quick.py           # E2E on Qwen2.5-0.5B (2 blocks)
│
├── external/                       # Cloned source repos (for reference)
│   ├── HAWQ-V3/                    # https://github.com/Zhen-Dong/HAWQ
│   ├── SliM-LLM/                   # https://github.com/Aaronhuang-778/SliM-LLM
│   ├── AutoRound/                  # https://github.com/intel/auto-round
│   └── SmoothQuant/                # https://github.com/mit-han-lab/smoothquant
│
├── paper/
│   ├── shmq_paper.pdf              # SHMQ paper
│   └── shmq_paper.txt              # Extracted text
│
├── download/                       # Output directory for quantized models
│
└── worklog.md → /home/z/my-project/worklog.md  # Detailed work log (1879 lines)
```

---

## Improvements over Original SHMQ

| Improvement | Source | Why Better |
|-------------|--------|------------|
| ILP bit allocation | HAWQ-V3 | Mathematically optimal (vs proportion mapping in Eq. 8) |
| AutoRound SignSGD | AutoRound | Optimizes rounding direction per-weight (vs RTN in original SHMQ) |
| SmoothQuant pre-processing | SmoothQuant | Eliminates activation outliers before quantization |
| SQC calibration | SliM-LLM | Salience-aware scale optimization (not in original SHMQ) |
| Fisher + PyHessian switch | — | Configurable inter-layer Hessian (default Fisher per paper) |

---

## What's NOT Included (and why)

| Item | Reason |
|------|--------|
| Marlin CUDA kernel | No GPU in dev env. Use AutoGPTQ's Marlin on user's GPU. |
| Real-time INT4 inference | We fake-quantize (dequantize to fp16). For real INT4, use AutoGPTQ packaging. |
| Perplexity / zero-shot eval | Requires full 128-sample calibration + GPU. Skeleton in `tests/test_e2e_quick.py`. |
| LoRA / fine-tuning integration | Out of scope (SHMQ paper mentions as future work). |

---

## Tested Models

| Model | Status | Notes |
|-------|--------|-------|
| GPT-2 small | ❌ Skipped | Uses Conv1D, not nn.Linear (would need adapter) |
| Qwen2.5-0.5B | ✅ E2E works | 2 transformer blocks, ~90s on CPU (full pipeline) |
| Qwen2.5-7B-Instruct | 🟡 Code-ready | Requires GPU (~16GB VRAM, ~7 min sensitivity per paper) |

---

## Key Equations (from SHMQ paper)

**Eq. 1** (Perturbation):
```
δL = (1/2) δW^T H δW
```

**Eq. 7** (Inter-layer Fisher sensitivity):
```
S^l_InterMQ = (1/2|D|) Σ_{d∈D} Σ_{i∈cout} (g_d^T δw^l_{i,:})^2
```

**Eq. 10** (Intra-layer OBS sensitivity):
```
S^l_{i,j} = (1/2)(w^l_{i,j} - Q(w^l_{i,j}))^2 / [(X X^T + λ·mean(diag(X X^T))·I)^{-1}]_{j,j}
```

**Eq. 11** (Manhattan norm):
```
S_IntraMQ_j = Σ_i∈cout |S^l_{i,j}|
```

**Eq. 12** (TopK identification):
```
Csen = I(S_IntraMQ, K),  K = ⌊cin · U_l⌉
```

**Decoupled permutation** (§3.2.3):
1. Sort channels ASCENDING by sensitivity → partition into Csen, Cinsen
2. Within Csen: sort by magnitude (minimize group-wise variance)
3. Within Cinsen: sort by magnitude
4. Final order = concat(Csen_sorted, Cinsen_sorted)

---

## License

Source code: MIT (inherits from HAWQ-V3, SliM-LLM, AutoRound, SmoothQuant — all MIT/Apache).
SHMQ paper: © ACL 2025 (cited for educational/research use).
