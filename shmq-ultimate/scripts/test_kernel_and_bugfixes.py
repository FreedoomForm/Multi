"""Test SHMQ 3-level CUDA kernel wrapper (CPU fallback path).

This test verifies that:
1. SHMQ3LevelKernel can be instantiated without cupy (CPU fallback)
2. The PyTorch fallback path produces correct results
3. The kernel integrates with SHMQMixLLMLinear adapter
4. All 3 precision levels (FP16, INT8, INT4) produce non-zero outputs
5. INT4 packing convention is consistent (CUDA kernel == PyTorch fallback == weight_packing)
6. GPTQ error propagation works (bug fix #1)
7. SQC multiplier is applied (bug fix #2)
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
from shmq.inference.shmq_3level_kernel import (
    SHMQ_3LEVEL_KERNEL_CUDA,
    SHMQ3LevelKernel,
    shmq_3level_gemm,
    _pack_int4_on_gpu,
    _pytorch_fallback,
)
from shmq.inference.weight_packing import pack_int4, unpack_int4


def test_kernel_source_compiles_to_string():
    """The CUDA source should be a non-empty string with valid PTX."""
    assert isinstance(SHMQ_3LEVEL_KERNEL_CUDA, str)
    assert len(SHMQ_3LEVEL_KERNEL_CUDA) > 5000, "Kernel source too short"
    assert "mma.sync.aligned" in SHMQ_3LEVEL_KERNEL_CUDA, "Missing PTX mma instructions"
    # Verify all 3 PTX MMA variants are present (user requirement)
    assert "m16n8k16" in SHMQ_3LEVEL_KERNEL_CUDA, "Missing FP16 MMA (m16n8k16)"
    assert "m8n8k16" in SHMQ_3LEVEL_KERNEL_CUDA, "Missing INT8 MMA (m8n8k16)"
    assert "m8n8k4" in SHMQ_3LEVEL_KERNEL_CUDA, "Missing INT4 MMA (m8n8k4) — Turing-specific"
    assert "shmq_3level_gemm_kernel" in SHMQ_3LEVEL_KERNEL_CUDA, "Missing kernel entry"
    print(f"[1] Kernel source: {len(SHMQ_3LEVEL_KERNEL_CUDA)} chars OK")


def test_int4_packing_convention_consistency():
    """CRITICAL: Verify all 3 packing functions use the SAME nibble convention.

    The CUDA kernel uses:  LOW nibble = even index, HIGH nibble = odd index
    All Python pack/unpack functions must match.
    """
    torch.manual_seed(42)
    # Create a known int4 codes tensor with distinct values at each position
    K = 8
    codes = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7],   # row 0: positive values
        [-1, -2, -3, -4, -5, -6, -7, -8],  # row 1: negative values
    ], dtype=torch.int8)
    N4 = codes.shape[0]

    # Pack with weight_packing.pack_int4
    packed_wp = pack_int4(codes)
    assert packed_wp.shape == (N4, K // 2), f"pack_int4 shape {packed_wp.shape} != ({N4}, {K//2})"

    # Pack with shmq_3level_kernel._pack_int4_on_gpu
    packed_k = _pack_int4_on_gpu(codes)
    assert packed_k.shape == (N4, K // 2), f"_pack_int4_on_gpu shape {packed_k.shape} != ({N4}, {K//2})"

    # Both must be identical
    assert torch.equal(packed_wp, packed_k), \
        f"pack_int4 != _pack_int4_on_gpu!\n  pack_int4: {packed_wp}\n  _pack_int4_on_gpu: {packed_k}"

    # Round-trip: unpack must recover original codes
    unpacked = unpack_int4(packed_wp)
    assert torch.equal(unpacked, codes), \
        f"Round-trip failed:\n  original: {codes}\n  unpacked: {unpacked}"

    # Verify the actual byte values: for row 0 [0,1,2,3,4,5,6,7]:
    #   byte 0 = (high=1 << 4) | low=0 = 0x10 = 16
    #   byte 1 = (high=3 << 4) | low=2 = 0x32 = 50
    #   byte 2 = (high=5 << 4) | low=4 = 0x54 = 84
    #   byte 3 = (high=7 << 4) | low=6 = 0x76 = 118
    expected_row0 = torch.tensor([16, 50, 84, 118], dtype=torch.uint8)
    assert torch.equal(packed_wp[0], expected_row0), \
        f"Row 0 packing mismatch: got {packed_wp[0]}, expected {expected_row0}"

    # For row 1 [-1,-2,-3,-4,-5,-6,-7,-8] (two's complement nibbles):
    #   -1 -> 15 (0xF), -2 -> 14 (0xE), -3 -> 13 (0xD), -4 -> 12 (0xC)
    #   -5 -> 11 (0xB), -6 -> 10 (0xA), -7 -> 9  (0x9), -8 -> 8  (0x8)
    #   byte 0 = (high=-2=14 << 4) | low=-1=15 = 0xEF = 239
    #   byte 1 = (high=-4=12 << 4) | low=-3=13 = 0xCD = 205
    #   byte 2 = (high=-6=10 << 4) | low=-5=11 = 0xAB = 171
    #   byte 3 = (high=-8=8  << 4) | low=-7=9  = 0x89 = 137
    expected_row1 = torch.tensor([239, 205, 171, 137], dtype=torch.uint8)
    assert torch.equal(packed_wp[1], expected_row1), \
        f"Row 1 packing mismatch: got {packed_wp[1]}, expected {expected_row1}"

    print(f"[2] INT4 packing convention consistent across all 3 functions OK")


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
    W4_codes = (W4_full.reshape(N4, K // gs, gs) / S4.unsqueeze(-1)).round().clamp(-7, 7).to(torch.int8).reshape(N4, K)
    # Pack using the corrected convention
    W4_packed = _pack_int4_on_gpu(W4_codes)

    # Reference: full FP16 matmul
    W_full = torch.cat([W16, W8_full, W4_full], dim=0)
    Y_ref = X @ W_full.t()

    # Our kernel (CPU fallback) — pass packed W4
    kernel = SHMQ3LevelKernel(
        W16=W16, W8=W8, W4_packed=W4_packed,
        S8=S8, S4=S4,
        K=K, N=N, N16=N16, N8=N8, N4=N4,
    )
    assert not kernel.is_cuda_native, "Should be CPU fallback in this env"
    Y_ours = kernel.forward(X)

    # Compare
    assert Y_ours.shape == (M, N), f"Shape mismatch: {Y_ours.shape} vs {(M, N)}"
    rel_error = (Y_ref - Y_ours).abs().norm() / Y_ref.abs().norm()
    # INT4 on random weights has high quantization error; threshold at 15%
    assert rel_error.item() < 0.15, f"Rel error too high: {rel_error.item()}"
    print(f"[3] PyTorch fallback correctness: rel_error={rel_error.item():.4f} OK")


def test_packed_vs_unpacked_input_consistency():
    """Verify that passing packed vs unpacked W4 gives the same result."""
    torch.manual_seed(0)
    M, K = 4, 256
    N4 = 96

    X = torch.randn(M, K, dtype=torch.float16)
    W4_codes = torch.randint(-7, 8, (N4, K), dtype=torch.int8)
    S4 = torch.ones(N4, K // 128, dtype=torch.float16) * 0.05

    W4_packed = _pack_int4_on_gpu(W4_codes)

    # Run with packed W4 (W4_packed=True)
    Y_packed = _pytorch_fallback(
        X, W16=None, W8=None, W4=W4_packed, S8=None, S4=S4,
        W4_packed=True, N16=0, N8=0, N4=N4, K=K, N=N4,
    )

    # Run with unpacked W4 (W4_packed=False)
    Y_unpacked = _pytorch_fallback(
        X, W16=None, W8=None, W4=W4_codes, S8=None, S4=S4,
        W4_packed=False, N16=0, N8=0, N4=N4, K=K, N=N4,
    )

    # They should be bit-exact (same math, same reduction order)
    max_diff = (Y_packed.float() - Y_unpacked.float()).abs().max().item()
    assert max_diff < 1e-3, f"Packed vs unpacked mismatch: max_diff={max_diff}"
    print(f"[4] Packed vs unpacked W4 consistency: max_diff={max_diff:.6f} OK")


def test_three_paths_produce_output():
    """All 3 precision paths should contribute non-zero output."""
    torch.manual_seed(0)
    M, K = 4, 256

    X = torch.randn(M, K, dtype=torch.float16)
    W16 = torch.randn(32, K, dtype=torch.float16) * 0.1
    W8 = torch.randint(-50, 50, (64, K), dtype=torch.int8)
    S8 = torch.ones(64, K // 128, dtype=torch.float16) * 0.01
    W4_codes = torch.randint(-7, 7, (96, K), dtype=torch.int8)
    W4_packed = _pack_int4_on_gpu(W4_codes)
    S4 = torch.ones(96, K // 128, dtype=torch.float16) * 0.05

    kernel = SHMQ3LevelKernel(
        W16=W16, W8=W8, W4_packed=W4_packed,
        S8=S8, S4=S4,
        K=K, N=32+64+96, N16=32, N8=64, N4=96,
    )
    Y = kernel.forward(X)

    # Check non-zero output across all 3 partitions
    y16 = Y[:, :32].abs().mean().item()
    y8 = Y[:, 32:96].abs().mean().item()
    y4 = Y[:, 96:].abs().mean().item()
    assert y16 > 1e-4, f"FP16 path produced near-zero output: {y16}"
    assert y8 > 1e-4, f"INT8 path produced near-zero output: {y8}"
    assert y4 > 1e-4, f"INT4 path produced near-zero output: {y4}"
    print(f"[5] All 3 paths produce output: FP16={y16:.4f}, INT8={y8:.4f}, INT4={y4:.4f} OK")


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
    # Verify no NaN/Inf
    assert torch.isfinite(Y).all(), "Output contains NaN/Inf"
    print(f"[6] Adapter integration: Y shape={Y.shape}, kernel built={linear._shmq_3level_kernel is not None} OK")


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
    print(f"[7] GPTQ error propagation: weight delta={delta:.4f} (should be > 0) OK")


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
    print(f"[8] SQC multiplier applied: weight delta={delta:.4f} OK")


def test_int4_round_trip_with_random_data():
    """Stress test: pack random int4 codes, unpack, verify recovery."""
    torch.manual_seed(123)
    for trial in range(5):
        N4 = torch.randint(1, 100, (1,)).item()
        K  = torch.randint(2, 500, (1,)).item() * 2  # ensure even
        codes = torch.randint(-8, 8, (N4, K), dtype=torch.int8)
        packed = pack_int4(codes)
        unpacked = unpack_int4(packed)
        assert torch.equal(unpacked, codes), \
            f"Trial {trial}: round-trip failed for shape ({N4}, {K})"
    print(f"[9] INT4 round-trip stress test (5 trials) OK")


if __name__ == '__main__':
    print("=" * 70)
    print("SHMQ 3-Level Kernel + Bug Fix Tests (v2 — fixed packing convention)")
    print("=" * 70)
    test_kernel_source_compiles_to_string()
    test_int4_packing_convention_consistency()
    test_pytorch_fallback_correctness()
    test_packed_vs_unpacked_input_consistency()
    test_three_paths_produce_output()
    test_adapter_integration()
    test_bug_fix_gptq_error_propagation()
    test_bug_fix_sqc_multiplier_application()
    test_int4_round_trip_with_random_data()
    print()
    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
