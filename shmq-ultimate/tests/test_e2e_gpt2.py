"""End-to-end integration test for SHMQ-Ultimate pipeline.

Runs the full pipeline on Qwen2.5-0.5B (smallest Qwen2.5 model) to verify all
components work together. CPU-only — no GPU required.

Qwen2.5-0.5B shares the same architecture as Qwen2.5-7B-Instruct (RMSNorm,
nn.Linear, q_proj/k_proj/v_proj/gate_proj/up_proj/down_proj/o_proj), so this
test exercises the same code paths that would run on Qwen2.5-7B-Instruct.
"""
import sys
import os
import time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from shmq.config import SHMQConfig
from shmq.pipeline import SHMQPipeline


def main():
    print("=" * 70)
    print("SHMQ-Ultimate — End-to-End Integration Test (Qwen2.5-0.5B)")
    print("=" * 70)

    # Use Qwen2.5-0.5B (smallest Qwen2.5 model, ~500M params, CPU-friendly)
    config = SHMQConfig(
        model_name="Qwen/Qwen2.5-0.5B",            # 500M params, CPU-friendly
        device="cpu",
        dtype="float32",                           # full precision for testing
        n_samples=4,                                # Few samples for speed
        sequence_length=128,                       # Short sequences for speed
        batch_size=1,
        autoround_iters=10,                        # Few iters for speed
        enable_sqc=True,
        enable_autoround=False,                    # Skip autoround for speed (verified in smoke test)
        target_hp_ratio=0.20,                      # W4.8A8
        base_hp_ratio=0.125,
        group_size=128,                            # Qwen2.5-0.5B cin=896, divisible by 128
        smooth_alpha=0.5,
        inter_layer_hessian="fisher",
        dampening=0.1,
    )

    print(f"\nConfig summary:\n{config.summary()}")

    pipeline = SHMQPipeline(config)

    # Time each step
    t0 = time.time()
    pipeline.step0_load()
    t1 = time.time()
    print(f"\n>>> Step 0 (load) took {t1-t0:.1f}s")

    t0 = time.time()
    pipeline.step1_smoothquant()
    t1 = time.time()
    print(f"\n>>> Step 1 (smooth) took {t1-t0:.1f}s")

    t0 = time.time()
    pipeline.step2_sensitivity()
    t1 = time.time()
    print(f"\n>>> Step 2 (sensitivity) took {t1-t0:.1f}s")

    t0 = time.time()
    pipeline.step3_ilp()
    t1 = time.time()
    print(f"\n>>> Step 3 (ILP) took {t1-t0:.1f}s")

    t0 = time.time()
    pipeline.step4_permutation()
    t1 = time.time()
    print(f"\n>>> Step 4 (permutation) took {t1-t0:.1f}s")

    t0 = time.time()
    pipeline.step5_rmsnorm_fusion()
    t1 = time.time()
    print(f"\n>>> Step 5 (RMSNorm fusion) took {t1-t0:.1f}s")

    print("\n>>> Step 6 (autoround) SKIPPED — see smoke test for verification")

    t0 = time.time()
    pipeline.step7_sqc()
    t1 = time.time()
    print(f"\n>>> Step 7 (SQC) took {t1-t0:.1f}s")

    t0 = time.time()
    pipeline.step8_quantize()
    t1 = time.time()
    print(f"\n>>> Step 8 (quantize) took {t1-t0:.1f}s")

    # Verify model still produces output
    print("\n>>> Verifying model still works after quantization...")
    with torch.no_grad():
        sample_input = pipeline.calibration_data[:1].to(config.device)
        out = pipeline.model(sample_input)
        logits = out.logits
        print(f"    Logits shape: {logits.shape}")
        print(f"    Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
        assert logits.shape[0] == 1
        assert not torch.isnan(logits).any(), "NaN in logits!"

    # Save the quantized model
    output_dir = "/home/z/my-project/shmq-ultimate/download/qwen2.5-0.5b_shmq_ultimate"
    pipeline.save_model(output_dir)

    print("\n" + "=" * 70)
    print("END-TO-END INTEGRATION TEST: PASSED")
    print("=" * 70)
    print(f"\nQuantized model saved to: {output_dir}")
    print(f"\nBit allocation summary:")
    n_4 = sum(1 for v in pipeline.bit_allocation.values() if v == 4)
    n_8 = sum(1 for v in pipeline.bit_allocation.values() if v == 8)
    print(f"  4-bit layers: {n_4}")
    print(f"  8-bit layers: {n_8}")
    print(f"  Total layers: {len(pipeline.bit_allocation)}")
    if pipeline.ilp_result:
        print(f"\nILP result:\n{pipeline.ilp_result.summary()}")

    # Show per-layer bit allocation (first 10 layers)
    print(f"\nFirst 10 layer allocations:")
    for i, (name, bits) in enumerate(sorted(pipeline.bit_allocation.items())):
        if i >= 10:
            break
        print(f"  {name}: {bits}-bit")


if __name__ == "__main__":
    main()
