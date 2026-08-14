"""SHMQ-Ultimate v2 — CPU smoke tests.

Verifies the core SHMQ math (permutation, RMSNorm fusion, sensitivity)
without requiring MixLLM or a real GPU. These tests can run on the dev
machine and validate the algorithm correctness before deployment to GPU.

Tests:
  1. Decoupled permutation produces valid permutation
  2. Decoupled permutation respects hp_ratio (K sensitive channels)
  3. K-axis weight permutation is reversible
  4. PermutedRMSNorm is mathematically equivalent to RMSNorm + permute
  5. Intra-layer sensitivity has correct shape and is non-negative
  6. Parallel constraint groups q/k/v correctly
  7. End-to-end: small model, FP32, CPU
"""
import sys
import os
import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from shmq_v2.config import SHMQv2Config
from shmq_v2.permutation.decoupled import (
    decoupled_permutation,
    apply_k_permutation_to_weight,
    apply_k_permutation_to_activation,
    compute_permutation_metric,
)
from shmq_v2.permutation.parallel import (
    get_parallel_group,
    group_linears_by_parallel_constraint,
    compute_group_sensitivity,
    assign_group_permutation,
)
from shmq_v2.permutation.rmsnorm_fusion import PermutedRMSNorm
from shmq_v2.sensitivity.intra_layer import (
    compute_intra_layer_sensitivity,
    compute_parallel_layer_sensitivity,
)
from shmq_v2.preprocessing.smoothquant import (
    compute_smooth_scales,
    apply_smoothquant_to_linear,
)


def test_decoupled_permutation_valid():
    """Permutation is a valid index permutation (no duplicates, covers all)."""
    sens = torch.randn(256)
    perm = decoupled_permutation(sens, hp_ratio=0.1, group_size=128)
    assert perm.shape == (256,)
    assert len(torch.unique(perm)) == 256, "Permutation has duplicates"
    assert perm.min() == 0 and perm.max() == 255
    print("  [PASS] decoupled_permutation produces valid permutation")


def test_decoupled_permutation_respects_hp_ratio():
    """K = round(cin * hp_ratio / group_size) * group_size sensitive channels."""
    sens = torch.randn(256)
    perm = decoupled_permutation(sens, hp_ratio=0.1, group_size=128)
    # First K channels should be the high-sensitivity ones
    K = 128 // 128 * 128  # round(256 * 0.1 / 128) * 128 = 0... hmm
    # Actually: 256 * 0.1 = 25.6, round(25.6 / 128) * 128 = 0 * 128 = 0
    # So K=0 when hp_ratio=0.1 and cin=256 (too small for group_size=128)
    # Use cin=1280 instead: 1280 * 0.1 = 128, round(128/128)*128 = 128
    sens = torch.randn(1280)
    perm = decoupled_permutation(sens, hp_ratio=0.1, group_size=128)
    K = 128
    # Top-K channels by sensitivity should be in perm[:K]
    top_k = torch.argsort(sens, descending=True)[:K]
    csen_set = set(perm[:K].tolist())
    top_k_set = set(top_k.tolist())
    assert csen_set == top_k_set, \
        f"Csen doesn't match top-K by sensitivity: diff={csen_set ^ top_k_set}"
    print("  [PASS] decoupled_permutation respects hp_ratio (K=128 sensitive channels)")


def test_k_permutation_reversible():
    """Applying permutation then inverse recovers original."""
    W = torch.randn(64, 128)
    perm = torch.randperm(128)
    W_perm = apply_k_permutation_to_weight(W, perm)
    # Inverse permutation
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(128)
    W_recovered = apply_k_permutation_to_weight(W_perm, inv_perm)
    assert torch.allclose(W, W_recovered, atol=1e-6), \
        f"Permutation not reversible: max diff = {(W - W_recovered).abs().max()}"
    print("  [PASS] K-axis weight permutation is reversible")


def test_permuted_rmsnorm_equivalence():
    """PermutedRMSNorm(x) == RMSNorm(x)[..., perm] * weight[perm].

    This is the core correctness property of the SHMQ §3.2.2 fusion.
    """
    hidden = 128
    batch = 4
    x = torch.randn(batch, hidden)
    perm = torch.randperm(hidden)

    # Reference: standard RMSNorm then permute
    weight = torch.ones(hidden)
    eps = 1e-6
    x_norm = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    y_ref = x_norm[..., perm] * weight[perm]

    # PermutedRMSNorm
    pn = PermutedRMSNorm(hidden_size=hidden, eps=eps, perm=perm)
    pn.weight.data.copy_(weight)
    y_test = pn(x)

    assert torch.allclose(y_ref, y_test, atol=1e-6), \
        f"PermutedRMSNorm mismatch: max diff = {(y_ref - y_test).abs().max()}"
    print("  [PASS] PermutedRMSNorm is mathematically equivalent to RMSNorm + permute")


def test_permuted_rmsnorm_with_real_weight():
    """PermutedRMSNorm with non-trivial weight vector still matches reference."""
    hidden = 64
    x = torch.randn(2, hidden)
    perm = torch.randperm(hidden)
    weight = torch.randn(hidden) * 0.1 + 1.0
    eps = 1e-5

    # Reference
    x_norm = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    y_ref = x_norm[..., perm] * weight[perm]

    # PermutedRMSNorm
    pn = PermutedRMSNorm(hidden_size=hidden, eps=eps, perm=perm)
    pn.weight.data.copy_(weight)
    y_test = pn(x)

    assert torch.allclose(y_ref, y_test, atol=1e-5), \
        f"Mismatch with real weight: max diff = {(y_ref - y_test).abs().max()}"
    print("  [PASS] PermutedRMSNorm matches with non-trivial weight vector")


def test_intra_sensitivity_shape():
    """Intra-layer sensitivity has correct shape and is non-negative."""
    weight = torch.randn(64, 256)
    activation = torch.randn(32, 256)
    sens = compute_intra_layer_sensitivity(
        weight, activation, lambda_damp=0.1, group_size=128
    )
    assert sens.shape == (256,), f"Wrong shape: {sens.shape}"
    assert (sens >= 0).all(), f"Sensitivity has negative values: min={sens.min()}"
    assert not sens.isnan().any(), "Sensitivity has NaN"
    print(f"  [PASS] Intra-layer sensitivity shape={sens.shape}, range=[{sens.min():.4f}, {sens.max():.4f}]")


def test_parallel_grouping():
    """Parallel constraint correctly groups q/k/v and gate/up."""
    names = [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
        "model.layers.0.mlp.down_proj",
        "model.layers.1.self_attn.q_proj",
        "model.layers.1.self_attn.k_proj",
        "model.layers.1.self_attn.v_proj",
    ]
    groups = group_linears_by_parallel_constraint(names)
    assert len(groups["0_attn_qkv"]) == 3, f"q/k/v should be 3 in group, got {len(groups['0_attn_qkv'])}"
    assert len(groups["0_mlp_gate_up"]) == 2, f"gate/up should be 2 in group"
    assert len(groups["1_attn_qkv"]) == 3, f"layer 1 q/k/v should be 3 in group"
    # o_proj and down_proj should be standalone
    standalone_keys = [k for k in groups if "standalone" in k]
    assert len(standalone_keys) >= 2, f"o_proj and down_proj should be standalone: {standalone_keys}"
    print(f"  [PASS] Parallel grouping: {len(groups)} groups, 2 standalone (o_proj, down_proj)")


def test_smoothquant_scales():
    """SmoothQuant scales have correct shape and positive values."""
    weight = torch.randn(128, 256) * 0.1
    activation = torch.randn(32, 256) * 5.0  # larger activations
    scales = compute_smooth_scales(weight, activation, alpha=0.5)
    assert scales.shape == (256,), f"Wrong shape: {scales.shape}"
    assert (scales > 0).all(), f"Scales should be positive: min={scales.min()}"
    # With alpha=0.5 and large activations, scales should be > 1 (migrate to weights)
    assert scales.mean() > 1.0, f"Expected scales > 1 for large activations, got mean={scales.mean()}"
    print(f"  [PASS] SmoothQuant scales: shape={scales.shape}, mean={scales.mean():.4f}")


def test_smoothquant_preserves_output():
    """SmoothQuant preserves the linear output: (X/s) @ (s*W)^T == X @ W^T."""
    torch.manual_seed(0)
    in_features, out_features = 128, 64
    linear = nn.Linear(in_features, out_features, bias=False)
    X = torch.randn(32, in_features)

    # Original output
    y_orig = linear(X)

    # Apply SmoothQuant
    scales = apply_smoothquant_to_linear(linear, X, alpha=0.5)
    # Smoothed output: (X / s) @ W_smooth^T
    X_smooth = X / scales
    y_smooth = linear(X_smooth)

    assert torch.allclose(y_orig, y_smooth, atol=1e-4), \
        f"SmoothQuant didn't preserve output: max diff = {(y_orig - y_smooth).abs().max()}"
    print("  [PASS] SmoothQuant preserves linear output")


def test_end_to_end_small():
    """End-to-end test: small synthetic weights, run permutation + RMSNorm fusion.

    Verifies that the SHMQ pipeline is mathematically consistent:
      1. Compute sensitivity
      2. Compute permutation
      3. Permute weight
      4. Replace RMSNorm with PermutedRMSNorm
      5. Forward pass with permuted input gives same output as unpermuted
    """
    torch.manual_seed(42)
    hidden = 128
    n_tokens = 16

    # Create a tiny "transformer block"
    rmsnorm = nn.RMSNorm(hidden) if hasattr(nn, "RMSNorm") else None
    if rmsnorm is None:
        # Manual RMSNorm for older PyTorch
        class RMSNorm(nn.Module):
            def __init__(self, h, eps=1e-6):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(h))
                self.eps = eps
            def forward(self, x):
                return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        rmsnorm = RMSNorm(hidden)

    linear = nn.Linear(hidden, hidden, bias=False)

    # Original input
    x = torch.randn(n_tokens, hidden)

    # Original forward
    x_norm = rmsnorm(x)
    y_orig = linear(x_norm)

    # === Apply SHMQ ===
    # 1. Compute sensitivity
    sens = compute_intra_layer_sensitivity(
        linear.weight.data, x, lambda_damp=0.1, group_size=128
    )

    # 2. Compute permutation
    perm = decoupled_permutation(sens, hp_ratio=0.1, group_size=128)

    # 3. Permute weight (W_perm = W[:, perm])
    W_perm = apply_k_permutation_to_weight(linear.weight.data, perm)

    # 4. Replace RMSNorm with PermutedRMSNorm
    pn = PermutedRMSNorm(hidden_size=hidden, eps=1e-6, perm=perm)
    pn.weight.data.copy_(rmsnorm.weight.data)

    # 5. New linear with permuted weight
    linear_perm = nn.Linear(hidden, hidden, bias=False)
    linear_perm.weight.data = W_perm

    # 6. Forward with PermutedRMSNorm
    x_norm_perm = pn(x)  # this outputs already-permuted activations
    y_shmq = linear_perm(x_norm_perm)

    # The output should match the original (mathematically equivalent)
    # Note: there's some FP precision difference due to different operation order
    max_diff = (y_orig - y_shmq).abs().max().item()
    print(f"  [PASS] End-to-end: max diff between original and SHMQ-pipeline = {max_diff:.6f}")
    assert max_diff < 1e-3, f"End-to-end diff too large: {max_diff}"


def test_config_validation():
    """SHMQv2Config validates correctly."""
    cfg = SHMQv2Config()
    cfg.validate()  # should not raise
    print(f"  [PASS] Config validates: {cfg.summary().split(chr(10))[1].strip()}")

    # Bad config: bit_percent doesn't sum to 100
    bad_cfg = SHMQv2Config(bit_percent={8: 20, 4: 90})
    try:
        bad_cfg.validate()
        assert False, "Should have raised"
    except AssertionError as e:
        if "sum to 100" in str(e):
            print("  [PASS] Config rejects bit_percent != 100")
        else:
            raise


def main():
    print("=" * 60)
    print("  SHMQ-Ultimate v2 — CPU Smoke Tests")
    print("=" * 60)
    tests = [
        test_decoupled_permutation_valid,
        test_decoupled_permutation_respects_hp_ratio,
        test_k_permutation_reversible,
        test_permuted_rmsnorm_equivalence,
        test_permuted_rmsnorm_with_real_weight,
        test_intra_sensitivity_shape,
        test_parallel_grouping,
        test_smoothquant_scales,
        test_smoothquant_preserves_output,
        test_end_to_end_small,
        test_config_validation,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    print("=" * 60)
    print(f"  Result: {passed} passed, {failed} failed")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
