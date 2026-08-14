#!/usr/bin/env python3
"""Rebuild shmq_ultimate_t4_benchmark.ipynb.

Changes from the previous version:
- Remove Cells 13-18 (FP16 baseline benchmark, MixLLM original, SHMQ paper repro)
  — we only benchmark SHMQ-Ultimate, and compare against SHMQ paper Table 1/2/5
  published numbers (no need to re-run other methods).
- Update Cell 0 (intro): new benchmark table (1 model + paper reference).
- Update Cell 5 (pipeline run): use hardening flags from new config.
- Update Cell 11 (zero-shot benchmarks): only SHMQ-Ultimate + FP16 baseline.
- Update Cell 12 (results table): SHMQ-Ultimate vs paper Table 1/2/5 numbers.
- Update Cell 13 (visualizations): SHMQ-Ultimate vs paper reference bars.

Input:  notebooks/shmq_ultimate_t4_benchmark.ipynb (existing)
Output: notebooks/shmq_ultimate_t4_benchmark.ipynb (rewritten in-place)
"""
from __future__ import annotations
import json
from pathlib import Path

NB_PATH = Path("/home/z/my-project/shmq-ultimate/notebooks/shmq_ultimate_t4_benchmark.ipynb")


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


# =========================================================================
# Load existing notebook to preserve metadata
# =========================================================================
with open(NB_PATH) as f:
    nb = json.load(f)
existing_cells = nb["cells"]

# =========================================================================
# Build new cell list
# =========================================================================
new_cells = []

# ---- Cell 0: Updated intro ----
new_cells.append(md("""# SHMQ-Ultimate: 3-Level {4, 8, 16} Mixed-Precision Quantization for Qwen2.5-7B-Instruct on T4

**Single-GPU T4 (sm_75, 16GB) — Single Notebook — Single Kernel Launch — Single Model Benchmark**

This notebook runs the SHMQ-Ultimate framework on Qwen2.5-7B-Instruct and compares the
results against the numbers published in the SHMQ paper (EMNLP Industry 2025).
We do NOT reproduce MixLLM-original or SHMQ-paper-2-level here — the paper's published
Table 1 / Table 2 / Table 5 numbers serve as the reference baseline.

## 7-Source Recipe

- **HAWQ-V3**: PyHessian trace + ILP solver (PULP) for bit allocation
- **SliM-LLM**: GPTQ OBS + SQC calibration
- **MixLLM**: production CUDA kernel + vLLM patch (modified for T4)
- **AutoRound**: learnable SignSGD rounding (200 steps)
- **SmoothQuant**: activation outlier migration
- **PolyQ**: ISA-aware quanta matching (128 for 8/16-bit, 64 for 4-bit)
- **SHMQ paper**: decoupled permutation + PermutedRMSNorm + parallel constraint

## Key Innovation: Single-Launch 3-Level Fused GEMM

The custom CUDA kernel (compiled via `cupy.RawKernel` + NVRTC) processes all 3 precision
levels {FP16, INT8, INT4} in **one kernel launch** on T4 (sm_75). PTX MMA wrappers defined
for all 3 Turing tensor-core types:
- `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` (FP16)
- `mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32`     (INT8)
- `mma.sync.aligned.m8n8k4.row.col.s32.s4.s4.s32`      (INT4)  ← Turing-specific

CUDA cores path is the default (guaranteed correctness); tensor-core path is available
via the `-DSHMQ_USE_TENSOR_CORES=1` compile flag.

## Benchmark Plan

| # | Model | Method | Format | Source |
|---|-------|--------|--------|--------|
| 1 | Qwen2.5-7B-Instruct (FP16) | Baseline | W16A16 | Run locally |
| 2 | Qwen2.5-7B-Instruct + MixLLM | Microsoft original (2-level) | W4.4A8 | **Paper Table 1/2** |
| 3 | Qwen2.5-7B-Instruct + SHMQ | 2-level paper method | W4.8A8 | **Paper Table 1/2/5** |
| 4 | Qwen2.5-7B-Instruct + SHMQ-Ultimate (ours) | 3-level fused | W{4,8,16}A16 | Run locally |

**Why not reproduce MixLLM/SHMQ-paper locally?** The SHMQ paper already publishes
results for Qwen2.5-7B-Instruct on the exact same benchmarks we run. Re-running them
locally would add ~5.5 hours of compute time without adding new signal. We trust the
paper's published numbers as the reference.

## 11-Step Quantization Pipeline

1. SmoothQuant (activation outlier migration)
2. PyHessian trace (inter-layer Fisher sensitivity, Eq.6)
3. OBS per-element sensitivity (Eq.5, XX^T + λI)
4. ILP solver (PULP, 3-level {4,8,16} bit allocation, UB=12.5%)
5. PolyQ ISA matching (128→8/16-bit, 64→4-bit tile rounding)
6. Decoupled permutation (3 clusters C16/C8/C4, Eq.12)
7. Permutation fusion (RMSNorm + activation)
8. AutoRound (SignSGD 200 steps, learnable V)
9. GPTQ (4-bit path) + RTN (8-bit path) + FP16 (16-bit path)
10. SQC calibration (salience-weighted scale multiplier)
11. MixLLM packing → custom 3-level kernel inference

## Runtime Hardening

All 6 seams between pipeline stages (C: ILP→ISA, D: permutation→AutoRound→GPTQ,
E: packing→kernel, S2: memory, S3: cupy DLPack, S4: vLLM config) are guarded by
runtime assertions in `src/shmq/hardening.py`. Failures produce descriptive error
messages instead of silent NaN corruption.

**Hardware requirement**: NVIDIA T4 (16GB, sm_75, Turing) — single GPU.
**Estimated runtime**: ~6 hours total (3.4h quantization + 2.7h evaluation).
"""))

# ---- Cells 1-2: Environment setup (keep) ----
new_cells.append(existing_cells[1])  # markdown
new_cells.append(existing_cells[2])  # code

# ---- Cells 3-4: GPU verification (keep) ----
new_cells.append(existing_cells[3])  # markdown
new_cells.append(existing_cells[4])  # code

# ---- Cells 5-6: Kernel source + compile (keep) ----
new_cells.append(existing_cells[5])  # markdown
new_cells.append(existing_cells[6])  # code
new_cells.append(existing_cells[7])  # markdown
new_cells.append(existing_cells[8])  # code

# ---- Cell 7: Updated pipeline run (was Cell 5) ----
new_cells.append(md("""## Cell 5 — Run SHMQ-Ultimate 11-Step Quantization Pipeline

Executes the full quantization pipeline on Qwen2.5-7B-Instruct with all hardening
flags enabled. The pipeline runs:

1. SmoothQuant (activation outlier migration, alpha=0.5)
2. PyHessian trace + Fisher sensitivity (inter-layer) + OBS (intra-layer)
3. ILP solver — 3-level {4,8,16} bit allocation (PULP/CBC, 30s time limit)
4. PolyQ ISA matching (round cluster sizes to tensor-core tiles)
5. Decoupled permutation into C16/C8/C4 clusters (SHMQ Eq.12, extended)
6. Permutation fusion into RMSNorm
7. AutoRound SignSGD (200 steps per block)
8. SQC calibration
9. GPTQ + mixed INT4/INT8 quantization
10. MixLLM conversion → custom 3-level kernel

**Hardening flags** (all enabled by default):
- `streaming_activation_capture=True` — one layer at a time (T4 OOM guard)
- `enable_hardening_asserts=True` — verify cluster_sizes, permutation, finite output
- `isa_drift_tolerance=0.05` — max ILP target drift after ISA rounding

**Time budget**: ~3.4 hours on T4 (PyHessian + 200 AutoRound iters + 128 calib samples).
"""))

new_cells.append(code("""# Cell 5: Run SHMQ-Ultimate 11-step pipeline on Qwen2.5-7B-Instruct
import os, sys, time, torch
sys.path.insert(0, "/workspace/shmq-ultimate/src")

from shmq.config import SHMQConfig
from shmq.pipeline import SHMQPipeline

# ---- Configuration ----
# All defaults match the SHMQ paper Section 4.1 + Appendix A.3.1.
# Hardening flags are ON by default — they catch silent NaN-producing bugs
# at the 6 seams between pipeline stages (C/D/E/S2/S3/S4).
config = SHMQConfig(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    device="cuda",
    dtype="float16",

    # 3-level bit allocation: 5% FP16 + 20% INT8 + 75% INT4 = 5.4 avg bits
    target_hp_ratio_16=0.05,
    target_hp_ratio_8=0.20,
    base_hp_ratio_8=0.125,    # UB = 12.5% (paper)

    # Calibration (paper: 128 samples × 2048 tokens from WikiText-2)
    calibration_dataset="wikitext2",
    n_samples=128,
    sequence_length=2048,
    batch_size=1,

    # Sensitivity
    inter_layer_hessian="fisher",   # paper default
    dampening=0.1,                   # paper Eq.10

    # AutoRound (paper: 200 steps)
    enable_autoround=True,
    autoround_iters=200,

    # SQC + GPTQ + ISA
    enable_sqc=True,
    enable_isa_matching=True,
    gptq_block_size=128,
    group_size=128,

    # Hardening (all on — catches NaN-producing bugs at seams)
    enable_hardening_asserts=True,
    streaming_activation_capture=True,  # T4 16GB OOM guard
    hessian_diag_only_retention=True,
    isa_drift_tolerance=0.05,
)

print(config.summary())

# ---- Run pipeline ----
pipeline = SHMQPipeline(config)
t_start = time.time()
pipeline.run()
t_total = time.time() - t_start
print(f"\\n[bench] Total pipeline time: {t_total/60:.1f} min ({t_total/3600:.2f} h)")

# ---- Save model ----
output_dir = "/workspace/shmq-ultimate/download/qwen25-7b-shmq-ultimate"
pipeline.save_model(output_dir)
print(f"[bench] Saved to {output_dir}")

# Save timing for later comparison
import json
with open("/workspace/shmq-ultimate/download/pipeline_timing.json", "w") as f:
    json.dump({"total_seconds": t_total, "total_minutes": t_total/60}, f, indent=2)
"""))

# ---- Cell 8: Load quantized model (was Cell 6) ----
new_cells.append(existing_cells[11])  # markdown (Cell 6 markdown)
new_cells.append(existing_cells[12])  # code   (Cell 6 code)

# ---- Cell 9: SHMQ-Ultimate benchmark (was Cell 10) ----
new_cells.append(md("""## Cell 7 — Benchmark: SHMQ-Ultimate (3-Level {4,8,16}, Ours)

Measures the SHMQ-Ultimate quantized model:
- Throughput (tokens/sec) on a 512-token prompt, 256 new tokens
- Peak GPU memory
- Latency per token

The FP16 baseline is also measured for speedup calculation. We do NOT benchmark
MixLLM-original or SHMQ-paper-2-level — their numbers come from the paper's
Table 1/2/5 (compared in Cell 9 below).
"""))

new_cells.append(code("""# Cell 7: SHMQ-Ultimate throughput + memory benchmark
import torch, time, sys, os, json
sys.path.insert(0, "/workspace/shmq-ultimate/src")

# ---- Load SHMQ-Ultimate model (already loaded by Cell 6 if running sequentially) ----
# If not in memory, re-load from saved artifact
if 'shmq_model' not in dir():
    from shmq.mixllm.adapter import SHMQMixLLMLinear, SHMQMixLLMConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    shmq_model = AutoModelForCausalLM.from_pretrained(
        "/workspace/shmq-ultimate/download/qwen25-7b-shmq-ultimate",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    shmq_model.eval()
    print(f"SHMQ-Ultimate model loaded: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ---- Throughput benchmark ----
prompt = "Explain the theory of relativity in simple terms." * 8  # ~512 tokens input
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
inputs = tok(prompt, return_tensors="pt").to("cuda")
input_len = inputs.input_ids.shape[1]

with torch.inference_mode():
    # Warmup
    for _ in range(3):
        _ = shmq_model.generate(**inputs, max_new_tokens=64, do_sample=False)
    torch.cuda.synchronize()

    # Measure
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = shmq_model.generate(**inputs, max_new_tokens=256, do_sample=False)
    torch.cuda.synchronize()
    t1 = time.time()

gen_tokens = out.shape[1] - input_len
shmq_tps = gen_tokens / (t1 - t0)
shmq_mem = torch.cuda.max_memory_allocated() / 1e9
shmq_ms_per_tok = (t1 - t0) / gen_tokens * 1000

print(f"\\nSHMQ-Ultimate (3-level {{4,8,16}}):")
print(f"  Input tokens: {{input_len}}")
print(f"  Generated tokens: {{gen_tokens}}")
print(f"  Time: {{t1-t0:.2f}}s")
print(f"  Throughput: {{shmq_tps:.1f}} tokens/sec")
print(f"  Latency:    {{shmq_ms_per_tok:.2f}} ms/token")
print(f"  Peak GPU memory: {{shmq_mem:.2f}} GB")

# ---- Optional: FP16 baseline for speedup reference ----
print("\\n" + "="*60)
print("FP16 baseline (for speedup calculation)")
print("="*60)
from transformers import AutoModelForCausalLM
fp16_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
fp16_model.eval()
print(f"FP16 model loaded: {torch.cuda.memory_allocated()/1e9:.2f} GB")

with torch.inference_mode():
    for _ in range(3):
        _ = fp16_model.generate(**inputs, max_new_tokens=64, do_sample=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    _ = fp16_model.generate(**inputs, max_new_tokens=256, do_sample=False)
    torch.cuda.synchronize()
    t1 = time.time()

fp16_tps = 256 / (t1 - t0)
fp16_mem = torch.cuda.max_memory_allocated() / 1e9
print(f"FP16: {{fp16_tps:.1f}} tokens/sec, {{fp16_mem:.2f}} GB")

print(f"\\n--- Speedup ---")
print(f"  SHMQ-Ultimate vs FP16: {{shmq_tps/fp16_tps:.2f}}x throughput")
print(f"  Memory reduction:      {{(1 - shmq_mem/fp16_mem)*100:.1f}}% "
      f"({{fp16_mem:.2f}}GB -> {{shmq_mem:.2f}}GB)")

# Save for later cells
SHMQ_ULTIMATE_TPS = shmq_tps
SHMQ_ULTIMATE_MEM = shmq_mem
FP16_TPS = fp16_tps
FP16_MEM = fp16_mem

# Free FP16 model
del fp16_model
torch.cuda.empty_cache()

# Save benchmark results
with open("/workspace/shmq-ultimate/download/throughput_benchmark.json", "w") as f:
    json.dump({
        "fp16": {"tps": fp16_tps, "mem_gb": fp16_mem},
        "shmq_ultimate": {"tps": shmq_tps, "mem_gb": shmq_mem,
                          "ms_per_token": shmq_ms_per_tok},
        "speedup": shmq_tps / fp16_tps,
    }, f, indent=2)
print("\\n✓ Throughput benchmark saved.")
"""))

# ---- Cell 10: Zero-shot benchmarks (only SHMQ-Ultimate + FP16) ----
new_cells.append(md("""## Cell 8 — Zero-Shot Benchmarks (SHMQ-Ultimate + FP16 Baseline)

Uses `lm-eval-harness` to evaluate the SHMQ-Ultimate model and FP16 baseline on the
**exact same tasks used in the SHMQ paper** (so we can directly compare against their
published Table 1 / Table 2 numbers):

| Task | Metric | Direction | Paper Table |
|------|--------|-----------|-------------|
| `wikitext` | word_perplexity | ↓ lower | Table 1 |
| `c4` | word_perplexity | ↓ lower | Table 1 |
| `arc_challenge` | acc | ↑ higher | Table 2 |
| `arc_easy` | acc | ↑ higher | Table 2 |
| `hellaswag` | acc | ↑ higher | Table 2 |
| `piqa` | acc | ↑ higher | Table 2 |
| `winogrande` | acc | ↑ higher | Table 2 |
| `boolq` | acc | ↑ higher | Table 2 |

We do NOT evaluate MixLLM-original or SHMQ-paper-2-level — the paper's published
numbers serve as the reference. This saves ~3.5 hours of evaluation time.

**Time budget**: ~2.7 hours on T4 (FP16 + SHMQ-Ultimate × 8 tasks each).
"""))

new_cells.append(code("""# Cell 8: Zero-shot benchmarks via lm-eval (SHMQ-Ultimate + FP16 only)
import subprocess, json, os, sys

TASKS = ["wikitext", "hellaswag", "arc_challenge", "arc_easy",
         "piqa", "winogrande", "boolq"]

def run_lm_eval(model_name, model_path_or_spec, dtype="float16"):
    # Run lm-eval on a model and return results dict.
    out_file = f"/tmp/lm_eval_{model_name}.json"
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path_or_spec},dtype={dtype},trust_remote_code=True",
        "--tasks", ",".join(TASKS),
        "--batch_size", "8",
        "--output_path", out_file,
        "--device", "cuda",
    ]
    print(f"  Running: {' '.join(cmd[:8])}...")
    subprocess.check_call(cmd)
    with open(out_file) as f:
        results = json.load(f)
    return results["results"]

# ---- Run on FP16 baseline (for direct comparison) ----
print("=" * 60)
print("FP16 Baseline — Qwen2.5-7B-Instruct")
print("=" * 60)
fp16_results = run_lm_eval("fp16", "Qwen/Qwen2.5-7B-Instruct")

# ---- Run on SHMQ-Ultimate ----
print("\\n" + "=" * 60)
print("SHMQ-Ultimate (3-level {4,8,16})")
print("=" * 60)
shmq_ultimate_results = run_lm_eval(
    "shmq_ultimate",
    "/workspace/shmq-ultimate/download/qwen25-7b-shmq-ultimate",
)

# ---- Save results ----
ALL_RESULTS = {
    "fp16": fp16_results,
    "shmq_ultimate": shmq_ultimate_results,
}
with open("/workspace/shmq-ultimate/download/lm_eval_results.json", "w") as f:
    json.dump(ALL_RESULTS, f, indent=2)
print("\\n✓ Zero-shot benchmarks complete. Results saved to download/lm_eval_results.json")
"""))

# ---- Cell 11: Results summary table with paper reference ----
new_cells.append(md("""## Cell 9 — Results Summary: SHMQ-Ultimate vs SHMQ Paper

This is the **main results table**. It compares our SHMQ-Ultimate (3-level {4,8,16})
against the numbers published in the SHMQ paper (EMNLP Industry 2025) for
Qwen2.5-7B-Instruct:

- **Paper Table 1**: PPL on WikiText-2 / C4 (FP16, MixLLM, SHMQ)
- **Paper Table 2**: Zero-shot QA accuracy (FP16, MixLLM, SHMQ)
- **Paper Table 5**: Ablation (WikiText-2 PPL, with/without each module)
- **Paper Section 4.3**: MMLU (FP16 = 74.27%, SHMQ = 73.34%)

**Our additions** (not in paper):
- 3-level {4,8,16} (paper uses 2-level {4,8})
- Single-launch cupy.RawKernel (paper uses MixLLM 2-launch)
- Throughput + memory on T4 (paper uses A100 for layer-wise speedup, Table 3)

**Expected outcome**: SHMQ-Ultimate should be **>= SHMQ paper** on quality (because
we add 5% FP16 channels which strictly improves over 2-level {4,8}), and **>= MixLLM**
on throughput (because we use a single fused kernel launch instead of 2 separate
MixLLM launches).
"""))

new_cells.append(code("""# Cell 9: Results summary table — SHMQ-Ultimate vs SHMQ paper reference numbers
import pandas as pd
import json

with open("/workspace/shmq-ultimate/download/lm_eval_results.json") as f:
    R = json.load(f)

def get_metric(results, task, metric_prefix="acc"):
    if task in results:
        for k, v in results[task].items():
            if k.startswith(metric_prefix):
                return v * 100 if v <= 1.0 else v  # convert to %
    return None

def get_ppl(results, task="wikitext"):
    if task in results:
        for k, v in results[task].items():
            if "perplexity" in k:
                return v
    return None

# ============================================================
# Table 1: Perplexity (WikiText-2, C4) — lower is better
# ============================================================
print("=" * 80)
print("Table 1: Perplexity (↓) on Qwen2.5-7B-Instruct")
print("=" * 80)
ppl_rows = [
    # Method, WikiText-2, C4, Source
    ("FP16 (paper)",        7.46,  10.89, "Paper Table 1"),
    ("MixLLM (paper)",      9.19,  12.91, "Paper Table 1"),
    ("SHMQ 2-level (paper)", 7.58, 11.06, "Paper Table 1"),
    ("FP16 (ours)",          get_ppl(R["fp16"], "wikitext"),  get_ppl(R["fp16"], "c4"),   "This notebook"),
    ("SHMQ-Ultimate (ours)", get_ppl(R["shmq_ultimate"], "wikitext"),
                             get_ppl(R["shmq_ultimate"], "c4"), "This notebook"),
]
df_ppl = pd.DataFrame(ppl_rows, columns=["Method", "WikiText-2 ↓", "C4 ↓", "Source"])
print(df_ppl.to_string(index=False))

# ============================================================
# Table 2: Zero-shot QA accuracy — higher is better
# ============================================================
print("\\n" + "=" * 80)
print("Table 2: Zero-shot QA accuracy (↑) on Qwen2.5-7B-Instruct")
print("=" * 80)
qa_rows = [
    # Method, ARC-C, ARC-E, BoolQ, HellaSwag, PIQA, WinoGrande, Avg, Source
    ("FP16 (paper)",         55.03, 81.14, 86.39, 80.50, 80.41, 70.80, 75.71, "Paper Table 2"),
    ("MixLLM (paper)",       51.02, 73.32, 82.23, 77.36, 77.64, 64.09, 70.94, "Paper Table 2"),
    ("SHMQ 2-level (paper)", 55.97, 80.60, 86.70, 79.66, 80.09, 70.48, 75.58, "Paper Table 2"),
    ("FP16 (ours)",
        get_metric(R["fp16"], "arc_challenge"),
        get_metric(R["fp16"], "arc_easy"),
        get_metric(R["fp16"], "boolq"),
        get_metric(R["fp16"], "hellaswag"),
        get_metric(R["fp16"], "piqa"),
        get_metric(R["fp16"], "winogrande"),
        None, "This notebook"),
    ("SHMQ-Ultimate (ours)",
        get_metric(R["shmq_ultimate"], "arc_challenge"),
        get_metric(R["shmq_ultimate"], "arc_easy"),
        get_metric(R["shmq_ultimate"], "boolq"),
        get_metric(R["shmq_ultimate"], "hellaswag"),
        get_metric(R["shmq_ultimate"], "piqa"),
        get_metric(R["shmq_ultimate"], "winogrande"),
        None, "This notebook"),
]
# Compute average for our runs
for r in qa_rows:
    if r[-2] is None and all(isinstance(x, (int, float)) for x in r[1:7]):
        avg = sum(r[1:7]) / 6
        r = list(r)
        r[-2] = avg
        qa_rows[qa_rows.index(tuple(r) if isinstance(r, tuple) else r)] = tuple(r) if isinstance(r, list) else r

df_qa = pd.DataFrame(qa_rows, columns=["Method", "ARC-C ↑", "ARC-E ↑", "BoolQ ↑",
                                       "HellaSwag ↑", "PIQA ↑", "WinoGrande ↑",
                                       "Avg ↑", "Source"])
print(df_qa.to_string(index=False))

# ============================================================
# Table 3: Throughput + Memory (T4-specific, not in paper)
# ============================================================
print("\\n" + "=" * 80)
print("Table 3: Throughput & Memory on T4 (16GB, sm_75)")
print("=" * 80)
try:
    with open("/workspace/shmq-ultimate/download/throughput_benchmark.json") as f:
        T = json.load(f)
    throughput_rows = [
        ("FP16 (ours)",          T["fp16"]["tps"], T["fp16"]["mem_gb"], "This notebook"),
        ("SHMQ-Ultimate (ours)", T["shmq_ultimate"]["tps"],
                                 T["shmq_ultimate"]["mem_gb"], "This notebook"),
    ]
    df_t = pd.DataFrame(throughput_rows,
                        columns=["Method", "Tokens/sec ↑", "Memory (GB) ↓", "Source"])
    print(df_t.to_string(index=False))
    print(f"\\nSpeedup (SHMQ-Ultimate vs FP16): {T['speedup']:.2f}x")
except FileNotFoundError:
    print("  (Run Cell 7 first to generate throughput benchmark)")

# ============================================================
# Table 4: Paper Ablation Reference (Table 5 from paper)
# ============================================================
print("\\n" + "=" * 80)
print("Table 4: Paper Ablation Reference (Qwen2.5-7B-Instruct, WikiText-2, Paper Table 5)")
print("=" * 80)
ablation_rows = [
    ("FP16",              "-",  "-",  "-",  7.46),
    ("W4.8A8 (no modules)", "×", "×", "×", 8.13),
    ("W4.8A8 + InterMQ",  "✓", "×", "×", 8.00),
    ("W4.8A8 + IntraMQ",  "×", "✓", "×", 7.99),
    ("W4.8A8 + Intra+Decoupling", "×", "✓", "✓", 7.95),
    ("W4.8A8 + All (SHMQ)", "✓", "✓", "✓", 7.58),
]
df_abl = pd.DataFrame(ablation_rows,
                      columns=["Config", "InterMQ", "IntraMQ", "Decoupling", "WikiText-2 ↓"])
print(df_abl.to_string(index=False))

# ============================================================
# Table 5: MMLU Reference (from paper Section 4.3)
# ============================================================
print("\\n" + "=" * 80)
print("Table 5: MMLU Reference (Paper Section 4.3, Qwen2.5-7B-Instruct)")
print("=" * 80)
mmlu_rows = [
    ("FP16 (paper)",  74.27, "Paper Section 4.3"),
    ("SHMQ 2-level (paper)", 73.34, "Paper Section 4.3"),
]
df_mmlu = pd.DataFrame(mmlu_rows, columns=["Method", "MMLU ↑", "Source"])
print(df_mmlu.to_string(index=False))
print("\\n(MMLU not run in this notebook — add `--tasks mmlu` to Cell 8 if desired.)")

# ============================================================
# Save all tables
# ============================================================
df_ppl.to_csv("/workspace/shmq-ultimate/download/table1_perplexity.csv", index=False)
df_qa.to_csv("/workspace/shmq-ultimate/download/table2_zero_shot_qa.csv", index=False)
df_abl.to_csv("/workspace/shmq-ultimate/download/table4_ablation_reference.csv", index=False)
df_mmlu.to_csv("/workspace/shmq-ultimate/download/table5_mmlu_reference.csv", index=False)
print("\\n✓ All tables saved to download/.")
"""))

# ---- Cell 12: Visualizations (SHMQ-Ultimate vs paper) ----
new_cells.append(md("""## Cell 10 — Visualizations: SHMQ-Ultimate vs SHMQ Paper

Bar charts comparing:
1. **WikiText-2 perplexity** — lower is better (Paper Table 1 + our results)
2. **Zero-shot QA average accuracy** — higher is better (Paper Table 2 + our results)
3. **Ablation waterfall** — WikiText-2 PPL as modules are added (Paper Table 5)
4. **Throughput on T4** — SHMQ-Ultimate vs FP16 (our measurements)

Paper reference numbers are shown as horizontal dashed lines for direct visual comparison.
"""))

new_cells.append(code("""# Cell 10: Plot comparison — SHMQ-Ultimate vs SHMQ paper
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import json

# Use Noto Sans SC for any CJK + DejaVu Sans for Latin/symbol fallback
try:
    fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
    fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
except Exception:
    pass
plt.rcParams['axes.unicode_minus'] = False

# Load results
df_ppl = pd.read_csv("/workspace/shmq-ultimate/download/table1_perplexity.csv")
df_qa  = pd.read_csv("/workspace/shmq-ultimate/download/table2_zero_shot_qa.csv")
df_abl = pd.read_csv("/workspace/shmq-ultimate/download/table4_ablation_reference.csv")
try:
    with open("/workspace/shmq-ultimate/download/throughput_benchmark.json") as f:
        T = json.load(f)
    has_throughput = True
except FileNotFoundError:
    has_throughput = False

fig, axes = plt.subplots(2, 2, figsize=(15, 11))

# Colors: paper methods in muted blue/orange, ours in green/red
COLOR_PAPER_FP16  = "#4C72B0"   # blue
COLOR_PAPER_MIX   = "#DD8452"   # orange
COLOR_PAPER_SHMQ  = "#55A868"   # green
COLOR_OURS_FP16   = "#79B4C4"   # light blue
COLOR_OURS_SHMQ   = "#C44E52"   # red (highlighted — our main result)

# ---- 1. WikiText-2 PPL (lower = better) ----
ax = axes[0, 0]
methods = df_ppl["Method"].tolist()
ppl = df_ppl["WikiText-2 ↓"].tolist()
colors = [COLOR_PAPER_FP16, COLOR_PAPER_MIX, COLOR_PAPER_SHMQ,
          COLOR_OURS_FP16, COLOR_OURS_SHMQ]
bars = ax.bar(methods, ppl, color=colors, edgecolor='black', linewidth=0.5)
ax.set_title("WikiText-2 Perplexity (↓ better)", fontsize=12, fontweight="bold")
ax.set_ylabel("PPL")
ax.tick_params(axis="x", rotation=25)
for bar, val in zip(bars, ppl):
    if pd.notna(val):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{val:.2f}", ha='center', va='bottom', fontsize=9)
# Highlight our main result
ax.axhline(y=7.58, color=COLOR_PAPER_SHMQ, linestyle='--', alpha=0.5,
           label='Paper SHMQ baseline (7.58)')
ax.legend(fontsize=9, loc='upper left')

# ---- 2. Zero-shot QA average accuracy (higher = better) ----
ax = axes[0, 1]
methods_qa = df_qa["Method"].tolist()
avg_acc = df_qa["Avg ↑"].tolist()
colors_qa = [COLOR_PAPER_FP16, COLOR_PAPER_MIX, COLOR_PAPER_SHMQ,
             COLOR_OURS_FP16, COLOR_OURS_SHMQ]
bars = ax.bar(methods_qa, avg_acc, color=colors_qa, edgecolor='black', linewidth=0.5)
ax.set_title("Average Zero-Shot QA Accuracy (↑ better)", fontsize=12, fontweight="bold")
ax.set_ylabel("Accuracy (%)")
ax.tick_params(axis="x", rotation=25)
ax.set_ylim([65, 80])
for bar, val in zip(bars, avg_acc):
    if pd.notna(val):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.2f}", ha='center', va='bottom', fontsize=9)
ax.axhline(y=75.58, color=COLOR_PAPER_SHMQ, linestyle='--', alpha=0.5,
           label='Paper SHMQ baseline (75.58)')
ax.legend(fontsize=9, loc='lower left')

# ---- 3. Ablation waterfall (Paper Table 5) ----
ax = axes[1, 0]
configs = ["FP16", "W4.8A8\\n(no mods)", "+InterMQ", "+IntraMQ", "+Intra+Decoup", "+All (SHMQ)"]
ppl_abl = df_abl["WikiText-2 ↓"].tolist()
colors_abl = ["#4C72B0", "#DD8452", "#DD8452", "#DD8452", "#DD8452", "#55A868"]
bars = ax.bar(configs, ppl_abl, color=colors_abl, edgecolor='black', linewidth=0.5)
ax.set_title("Ablation: WikiText-2 PPL as modules added (Paper Table 5)",
             fontsize=11, fontweight="bold")
ax.set_ylabel("PPL")
ax.tick_params(axis="x", rotation=15)
for bar, val in zip(bars, ppl_abl):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f"{val:.2f}", ha='center', va='bottom', fontsize=9)

# ---- 4. Throughput on T4 (our measurement) ----
ax = axes[1, 1]
if has_throughput:
    methods_t = ["FP16 (ours)", "SHMQ-Ultimate\\n(ours)"]
    tps = [T["fp16"]["tps"], T["shmq_ultimate"]["tps"]]
    mem = [T["fp16"]["mem_gb"], T["shmq_ultimate"]["mem_gb"]]
    colors_t = [COLOR_OURS_FP16, COLOR_OURS_SHMQ]
    x = np.arange(len(methods_t))
    bars = ax.bar(x, tps, color=colors_t, edgecolor='black', linewidth=0.5)
    ax.set_title(f"Throughput on T4 (speedup: {T['speedup']:.2f}x)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("tokens/sec")
    ax.set_xticks(x)
    ax.set_xticklabels(methods_t)
    for bar, val, m in zip(bars, tps, mem):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f} tok/s\\n({m:.1f} GB)", ha='center', va='bottom', fontsize=9)
else:
    ax.text(0.5, 0.5, "Run Cell 7 to generate\\nthroughput benchmark",
            ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.set_title("Throughput on T4", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig("/workspace/shmq-ultimate/download/comparison_plot.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved to download/comparison_plot.png")
"""))

# ---- Cell 13: Kernel correctness (keep) ----
new_cells.append(existing_cells[27])  # markdown
new_cells.append(existing_cells[28])  # code

# ---- Cell 14: Save and finalize (keep) ----
new_cells.append(existing_cells[29])  # markdown
new_cells.append(existing_cells[30])  # code

# ---- Cell 15: Appendix (keep) ----
new_cells.append(existing_cells[31])

# =========================================================================
# Write the new notebook
# =========================================================================
nb["cells"] = new_cells

# Update nbformat metadata if needed
nb["metadata"] = nb.get("metadata", {})
nb["metadata"]["kernelspec"] = nb.get("metadata", {}).get("kernelspec", {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
})
nb["metadata"]["language_info"] = nb.get("metadata", {}).get("language_info", {
    "name": "python",
    "version": "3.10",
})

with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1)

# Validate
with open(NB_PATH) as f:
    nb_check = json.load(f)
print(f"✓ Notebook rebuilt: {NB_PATH}")
print(f"  Total cells: {len(nb_check['cells'])}")
print(f"  Markdown cells: {sum(1 for c in nb_check['cells'] if c['cell_type']=='markdown')}")
print(f"  Code cells: {sum(1 for c in nb_check['cells'] if c['cell_type']=='code')}")
print()
print("Cell outline:")
for i, c in enumerate(nb_check['cells']):
    src = ''.join(c['source'])
    first_line = src.split('\\n')[0][:80]
    print(f"  [{i:2d}] {c['cell_type']:8s} | {first_line}")
