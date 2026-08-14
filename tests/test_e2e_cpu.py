"""End-to-end CPU test: full 11-step SHMQ-Ultimate pipeline.

The sandbox has ~1 GB RAM, so instead of downloading a pretrained LLM we
build a tiny random-initialised Llama-architecture model (same module names
as Qwen/Llama: q/k/v/o_proj, gate/up/down_proj, input_layernorm,
post_attention_layernorm) and run the complete pipeline on it.  This
exercises every step end-to-end:
  SmoothQuant -> XX^T capture -> Fisher sensitivity -> ILP {16,8,4} ->
  OBS channel sensitivity -> decoupled 3-cluster permutation -> RMSNorm
  fusion -> AutoRound -> SQC -> mixed-bit GPTQ -> packed SHMQUltimateLinear.

Checks: bit budget, PPL sanity, packed-vs-dequant logits equivalence,
memory compression.  (Real-model quality numbers come from the GPU run.)
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from shmq_ultimate import SHMQUltimateConfig, SHMQUltimatePipeline
from shmq_ultimate.calibration import get_calibration_batches
from shmq_ultimate.evaluation import wikitext2_perplexity, linear_weight_bytes

VOCAB = 8000  # small vocab keeps logits + backward within the 1 GB sandbox


class ClampTokenizer:
    """Wraps a real tokenizer, folding token ids into a small vocab."""

    def __init__(self, tok, vocab: int):
        self.tok = tok
        self.vocab = vocab

    def __call__(self, text, return_tensors=None):
        enc = self.tok(text, return_tensors=return_tensors)
        enc["input_ids"] = enc["input_ids"] % self.vocab
        return enc

    def __len__(self):
        return self.vocab


def build_tiny_model(vocab_size: int):
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=1024,
        rms_norm_eps=1e-6,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg)
    model.eval()
    return model


def main():
    torch.manual_seed(0)
    from transformers import AutoTokenizer
    tokenizer = ClampTokenizer(AutoTokenizer.from_pretrained("gpt2"), VOCAB)
    model = build_tiny_model(VOCAB)

    cfg = SHMQUltimateConfig(
        model_name="tiny-llama-random",
        device="cpu", dtype="float32",
        n_samples=16, sequence_length=256,
        target_avg_bits=4.8,
        # small dims -> use 32-wide groups/quanta so clusters stay aligned
        group_size=32, quanta_int8=32, quanta_int4=32,
        autoround_iters=40, sqc_grid=10,
    )

    import copy
    fp_model = copy.deepcopy(model)

    pipe = SHMQUltimatePipeline(cfg)
    t0 = time.time()
    model = pipe.run(model=model, tokenizer=tokenizer)

    # ---- quality: perplexity before/after ----------------------------------
    text = " ".join(["The quick brown fox jumps over the lazy dog."] * 800)
    ids = tokenizer(text, return_tensors="pt")["input_ids"]

    ppl_fp = wikitext2_perplexity(fp_model, ids, seq_len=256, max_windows=4)
    ppl_q = wikitext2_perplexity(model, ids, seq_len=256, max_windows=4)
    avg_bits = pipe.report["avg_bits"]
    print(f"\nFP32 ppl (tiny random model): {ppl_fp:.1f}")
    print(f"SHMQ-Ultimate ppl:            {ppl_q:.1f}")
    print(f"avg weight bits: {avg_bits:.3f} (target {cfg.target_avg_bits})")

    assert avg_bits <= cfg.target_avg_bits + 1e-6, "bit budget violated"
    assert ppl_q == ppl_q, "quantized ppl is NaN"
    assert ppl_q < ppl_fp * 2.0, f"ppl degradation too large: {ppl_q} vs {ppl_fp}"

    # ---- permutation coverage ------------------------------------------------
    n_perm = sum(1 for p in pipe.partitions.values() if not p.is_identity())
    print(f"permuted layers: {n_perm}")
    assert n_perm > 0, "no layer received a non-identity permutation"

    # ---- packed inference conversion ----------------------------------------
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
    print(f"linear weight memory: fp16 {fp16_bytes/1e6:.2f} MB -> "
          f"packed {mem_after/1e6:.2f} MB ({comp:.2f}x compression)")

    pipe.save("quantized_models/e2e_cpu_test")

    result = {
        "model": "tiny-llama-random (4 blocks, hidden 256)",
        "ppl_fp32": ppl_fp, "ppl_shmq_ultimate": ppl_q,
        "avg_bits": avg_bits, "packed_rel_err": rel,
        "permuted_layers": n_perm,
        "compression_vs_fp16": comp,
        "elapsed_s": time.time() - t0,
    }
    os.makedirs("benchmarks/results", exist_ok=True)
    with open("benchmarks/results/e2e_cpu_pipeline.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nE2E CPU TEST PASSED in {result['elapsed_s']:.0f}s")


if __name__ == "__main__":
    main()
