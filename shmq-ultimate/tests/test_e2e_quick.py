"""Quick end-to-end test for SHMQ-Ultimate pipeline on Qwen2.5-0.5B.

Processes only ONE transformer block (instead of all 24) to verify the pipeline
works end-to-end on CPU within a reasonable time.
"""
import sys
import os
import time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from shmq.config import SHMQConfig
from shmq.pipeline import SHMQPipeline
from shmq.utils import get_module_by_name


def main():
    print("=" * 70)
    print("SHMQ-Ultimate — Quick E2E Test (Qwen2.5-0.5B, 1 transformer block)")
    print("=" * 70)

    config = SHMQConfig(
        model_name="Qwen/Qwen2.5-0.5B",
        device="cpu",
        dtype="float32",
        n_samples=4,
        sequence_length=128,
        batch_size=1,
        autoround_iters=5,
        enable_sqc=True,
        enable_autoround=False,
        target_hp_ratio=0.20,
        base_hp_ratio=0.125,
        group_size=128,
        smooth_alpha=0.5,
        inter_layer_hessian="fisher",
        dampening=0.1,
    )

    pipeline = SHMQPipeline(config)

    # Step 0
    t0 = time.time()
    pipeline.step0_load()
    print(f">>> Step 0 took {time.time()-t0:.1f}s")

    # Limit to first 2 transformer blocks (14 layers: 7 per block × 2)
    # 7 layers per block: q, k, v, o, gate, up, down
    pipeline.layer_infos = [l for l in pipeline.layer_infos if l.block_idx in (0, 1)]
    pipeline.layer_names = [l.name for l in pipeline.layer_infos]
    pipeline.parallel_groups = pipeline._build_parallel_groups()
    print(f"\n[limit] Restricting to {len(pipeline.layer_names)} layers (2 blocks)")

    # Step 1
    t0 = time.time()
    pipeline.step1_smoothquant()
    print(f">>> Step 1 took {time.time()-t0:.1f}s")

    # Step 2
    t0 = time.time()
    pipeline.step2_sensitivity()
    print(f">>> Step 2 took {time.time()-t0:.1f}s")

    # Step 3
    t0 = time.time()
    pipeline.step3_ilp()
    print(f">>> Step 3 took {time.time()-t0:.1f}s")

    # Step 4
    t0 = time.time()
    pipeline.step4_permutation()
    print(f">>> Step 4 took {time.time()-t0:.1f}s")

    # Step 5
    t0 = time.time()
    pipeline.step5_rmsnorm_fusion()
    print(f">>> Step 5 took {time.time()-t0:.1f}s")

    # Step 7
    t0 = time.time()
    pipeline.step7_sqc()
    print(f">>> Step 7 took {time.time()-t0:.1f}s")

    # Step 8
    t0 = time.time()
    pipeline.step8_quantize()
    print(f">>> Step 8 took {time.time()-t0:.1f}s")

    # Verify
    print("\n>>> Verifying model still produces output...")
    with torch.no_grad():
        sample_input = pipeline.calibration_data[:1].to(config.device)
        out = pipeline.model(sample_input)
        logits = out.logits
        print(f"    Logits shape: {logits.shape}")
        print(f"    Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
        assert not torch.isnan(logits).any(), "NaN in logits!"
        assert not torch.isinf(logits).any(), "Inf in logits!"

    # Bit allocation
    n_4 = sum(1 for v in pipeline.bit_allocation.values() if v == 4)
    n_8 = sum(1 for v in pipeline.bit_allocation.values() if v == 8)
    print(f"\nBit allocation: {n_4} layers @ 4-bit, {n_8} layers @ 8-bit")
    print(f"\nFirst 10 layer allocations:")
    for i, (name, bits) in enumerate(sorted(pipeline.bit_allocation.items())):
        if i >= 10:
            break
        print(f"  {name}: {bits}-bit")

    print("\n" + "=" * 70)
    print("QUICK E2E TEST: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
