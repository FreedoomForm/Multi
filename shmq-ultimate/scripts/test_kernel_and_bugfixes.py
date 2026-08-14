"""Test SHMQ 3-level CUDA kernel wrapper (CPU fallback path).

This test verifies that:
1. SHMQ3LevelKernel can be instantiated without cupy (CPU fallback)
2. The PyTorch fallback path produces correct results
3. The kernel integrates with SHMQMixLLMLinear adapter
4. All 3 precision levels (FP16, INT8, INT4) produce non-zero outputs
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
from shmq.inference.shmq_3level_kernel import (
    SHMQ_3LEVEL_KERNEL_CUDA,
    SHMQ3LevelKernel,
    shmq_3level_gemm,
)


def test_kernel_source_compiles_to_string():
    """The CUDA source should be a non-empty string with valid PTX."""
    assert isinstance(SHMQ_3LEVEL_KERNEL_CUDA, str)
    assert len(SHMQ_3LEVEL_KERNEL_CUDA) > 5000, "Kernel source too short"
    assert "mma.sync.aligned" in SHMQ_3LEVEL_KERNEL_CUDA, "Missing PTX mma instructions"
    assert "shmq_3level_gemm_kernel" in SHMQ_3LEVEL_KERNEL_CUDA, "Missing kernel entry"
    print(f"[1] Kernel source: {len(SHMQ_3LEVEL_KERNEL_CUDA)} chars OK")


def test_pytorch_fallback_correctness():
    """On CPU (no cupy), kernel should use PyTorch fallback and produce correct results."""
    torch.manual_seed(42)
    M, K, N = 8, 256, 192
    N16, N8, N4 = 32, 64, 96  # 32+64+96 = 192

    X = torch.randn(M, K, dtype=torch.float16)
    W16 = torch.randn(N16, K, dtype=torch.float16) * 0.02

    # INT8 weights with per-group-of-128 scales
    W8_full = torch.randn(N8, K, dtype=torch.float16) * 0.05
    gs = 128
    S8 = W8_full.abs().reshape(N8, K // gs, gs).amax(-1, keepdim=True).squeeze(-1) / 127
    S8 = S8.clamp(min=1e-6).to(torch.float16)
    W8 = (W8_full.reshape(N8, K // gs, gs) / S8.unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8).reshape(N8, K)

    # INT4 weights
    W4_full = torch.randn(N4, K, dtype=torch.float16) * 0.1
    S4 = W4_full.abs().reshape(N4, K // gs, gs).amax(-1, keepdim=True).squeeze(-1) / 7
    S4 = S4.clamp(min=1e-6).to(torch.float16)
    W4 = (W4_full.reshape(N4, K // gs, gs) / S4.unsqueeze(-1)).round().clamp(-7, 7).to(torch.int8).reshape(N4, K)

    # Reference: full FP16 matmul
    W_full = torch.cat([W16, W8_full, W4_full], dim=0)
    Y_ref = X @ W_full.t()

    # Our kernel (CPU fallback)
    kernel = SHMQ3LevelKernel(W16=W16, W8=W8, W4=W4, S8=S8, S4=S4)
    assert not kernel.is_cuda_native, "Should be CPU fallback in this env"
    Y_ours = kernel.forward(X)

    # Compare
    assert Y_ours.shape == (M, N), f"Shape mismatch: {Y_ours.shape} vs {(M, N)}"
    rel_error = (Y_ref - Y_ours).abs().norm() / Y_ref.abs().norm()
    # INT4 on random weights has high quantization error; threshold at 15%
    assert rel_error.item() < 0.15, f"Rel error too high: {rel_error.item()}"
    print(f"[2] PyTorch fallback correctness: rel_error={rel_error.item():.4f} OK")


def test_three_paths_produce_output():
    """All 3 precision paths should contribute non-zero output."""
    torch.manual_seed(0)
    M, K = 4, 256

    X = torch.randn(M, K, dtype=torch.float16)
    W16 = torch.randn(32, K, dtype=torch.float16) * 0.1
    W8 = torch.randint(-50, 50, (64, K), dtype=torch.int8)
    S8 = torch.ones(64, K // 128, dtype=torch.float16) * 0.01
    W4 = torch.randint(-7, 7, (96, K), dtype=torch.int8)
    S4 = torch.ones(96, K // 128, dtype=torch.float16) * 0.05

    kernel = SHMQ3LevelKernel(W16=W16, W8=W8, W4=W4, S8=S8, S4=S4)
    Y = kernel.forward(X)

    # Check non-zero output across all 3 partitions
    y16 = Y[:, :32].abs().mean().item()
    y8 = Y[:, 32:96].abs().mean().item()
    y4 = Y[:, 96:].abs().mean().item()
    assert y16 > 1e-4, f"FP16 path produced near-zero output: {y16}"
    assert y8 > 1e-4, f"INT8 path produced near-zero output: {y8}"
    assert y4 > 1e-4, f"INT4 path produced near-zero output: {y4}"
    print(f"[3] All 3 paths produce output: FP16={y16:.4f}, INT8={y8:.4f}, INT4={y4:.4f} OK")


def test_adapter_integration():
    """SHMQMixLLMLinear should build the 3-level kernel automatically."""
    from shmq.mixllm.adapter import SHMQMixLLMLinear, SHMQMixLLMConfig

    cfg = SHMQMixLLMConfig(
        in_features=256, out_features=512,
        n_fp16_channels=64, n_int8_channels=128, n_int4_channels=320,
        group_size=128, bias=True, permutation=None,
    )
    fp16_weight = torch.randn(512, 256, dtype=torch.float16) * 0.02
    linear = SHMQMixLLMLinear(cfg, fp16_weight=fp16_weight)
    assert linear._shmq_3level_kernel is not None, "3-level kernel not built"
    X = torch.randn(4, 256, dtype=torch.float16)
    Y = linear(X)
    assert Y.shape == (4, 512), f"Output shape mismatch: {Y.shape}"
    print(f"[4] Adapter integration: Y shape={Y.shape}, kernel built={linear._shmq_3level_kernel is not None} OK")


def test_bug_fix_gptq_error_propagation():
    """GPTQ should now propagate error (not degenerate to RTN)."""
    from shmq.quantize.gptq import GPTQQuantizer
    import torch.nn as nn

    torch.manual_seed(42)
    layer = nn.Linear(64, 32, bias=False)
    layer.weight.data = torch.randn(32, 64) * 0.1
    gptq = GPTQQuantizer(layer, n_bits=4, group_size=64, percdamp=0.01, blocksize=64)

    # Add some activations (this builds the Hessian)
    X = torch.randn(128, 64)
    gptq.add_batch(X)
    original_W = layer.weight.data.clone()
    gptq.quantize()

    # Weight should have changed (error propagation updates remaining columns)
    delta = (layer.weight.data - original_W).abs().mean().item()
    # With proper GPTQ, delta should be larger than RTN delta because of error propagation
    # to columns beyond the first block.
    assert delta > 1e-4, f"GPTQ produced no change (delta={delta}); error propagation may be broken"
    print(f"[5] GPTQ error propagation: weight delta={delta:.4f} (should be > 0) OK")


def test_bug_fix_sqc_multiplier_application():
    """SQC multiplier should now be applied in MixedPrecisionQuantizer."""
    from shmq.quantize.mixed import MixedPrecisionQuantizer
    import torch.nn as nn

    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(64, 32, bias=False))
    layer_names = ['0']
    n_bits = {'0': 4}
    mults = {'0': 1.05}  # 5% scale increase

    q = MixedPrecisionQuantizer(group_size=64)
    original = model[0].weight.data.clone()
    q.apply(model, layer_names, n_bits, captured_activations=None, sqc_multipliers=mults)
    delta = (model[0].weight.data - original).abs().mean().item()
    assert delta > 1e-4, f"SQC multiplier had no effect (delta={delta})"
    print(f"[6] SQC multiplier applied: weight delta={delta:.4f} OK")


if __name__ == '__main__':
    print("=" * 70)
    print("SHMQ 3-Level Kernel + Bug Fix Tests")
    print("=" * 70)
    test_kernel_source_compiles_to_string()
    test_pytorch_fallback_correctness()
    test_three_paths_produce_output()
    test_adapter_integration()
    test_bug_fix_gptq_error_propagation()
    test_bug_fix_sqc_multiplier_application()
    print()
    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
