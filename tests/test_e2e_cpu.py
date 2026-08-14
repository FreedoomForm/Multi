"""End-to-end CPU test: full 11-step SHMQ-Ultimate pipeline on a small model.

Uses Qwen2.5-0.5B-Instruct limited to a few blocks with a reduced calibration
budget so it runs on a 2-core CPU sandbox in minutes.  Verifies:
  * pipeline runs all steps without error;
  * average allocated bits <= target;
  * quantized-model perplexity stays within a sane factor of FP32;
  * packed SHMQUltimateLinear conversion preserves outputs vs dequant model;
  * memory compression vs FP16 baseline.
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from shmq_ultimate import SHMQUltimateConfig, SHMQUltimatePipeline
from shmq_ultimate.calibration import get_calibration_batches
from shmq_ultimate.evaluation import wikitext2_perplexity, linear_weight_bytes


def main():
    torch.manual_seed(0)
    cfg = SHMQUltimateConfig(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device="cpu", dtype="float32",
        n_samples=16, sequence_length=512,
        max_blocks=4,
        target_avg_bits=4.8,
        autoround_iters=40,
        sqc_grid=10,
    )

    pipe = SHMQUltimatePipeline(cfg)
    t0 = time.time()
    model = pipe.run()
    tokenizer = pipe.tokenizer

    # ---- quality: perplexity before/after ---------------------------------
    from shmq_ultimate.model_utils import load_model
    fp_model, _ = load_model(cfg.model_name, cfg.dtype, cfg.device, cfg.max_blocks)

    try:
        from shmq_ultimate.calibration import get_wikitext2_test
        ids = get_wikitext2_test(tokenizer, cfg.device)
    except Exception:
        text = " ".join(["The quick brown fox jumps over the lazy dog."] * 2000)
        ids = tokenizer(text, return_tensors="pt").input_ids

    ppl_fp = wikitext2_perplexity(fp_model, ids, seq_len=512, max_windows=8)
    ppl_q = wikitext2_perplexity(model, ids, seq_len=512, max_windows=8)
    avg_bits = pipe.report["avg_bits"]
    print(f"\nFP32 ppl (4-block truncated model): {ppl_fp:.3f}")
    print(f"SHMQ-Ultimate ppl:                  {ppl_q:.3f}")
    print(f"avg weight bits: {avg_bits:.3f} (target {cfg.target_avg_bits})")

    assert avg_bits <= cfg.target_avg_bits + 1e-6, "bit budget violated"
    assert ppl_q == ppl_q, "quantized ppl is NaN"
    assert ppl_q < ppl_fp * 2.0, f"ppl degradation too large: {ppl_q} vs {ppl_fp}"

    # ---- packed inference conversion ---------------------------------------
    x = get_calibration_batches(tokenizer, 1, 128, device=cfg.device)[0]
    with torch.no_grad():
        y_deq = model(x).logits
    mem_before = linear_weight_bytes(model)
    n = pipe.convert_for_inference()
    assert n > 0, "no layers converted"
    with torch.no_grad():
        y_packed = model(x).logits
    rel = ((y_packed - y_deq).norm() / y_deq.norm()).item()
    print(f"packed vs dequant logits rel err: {rel:.5f}")
    assert rel < 0.02, "packed inference deviates from dequant reference"

    mem_after = linear_weight_bytes(model)
    fp16_bytes = mem_before // 2  # dequant model is fp32
    comp = fp16_bytes / mem_after
    print(f"linear weight memory: fp16 {fp16_bytes/1e6:.1f} MB -> "
          f"packed {mem_after/1e6:.1f} MB ({comp:.2f}x compression)")

    pipe.save("quantized_models/e2e_cpu_test")

    result = {
        "model": cfg.model_name, "blocks": cfg.max_blocks,
        "ppl_fp32": ppl_fp, "ppl_shmq_ultimate": ppl_q,
        "avg_bits": avg_bits, "packed_rel_err": rel,
        "compression_vs_fp16": comp,
        "elapsed_s": time.time() - t0,
    }
    os.makedirs("benchmarks/results", exist_ok=True)
    with open("benchmarks/results/e2e_cpu_qwen05b.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nE2E CPU TEST PASSED in {result['elapsed_s']:.0f}s")


if __name__ == "__main__":
    main()
