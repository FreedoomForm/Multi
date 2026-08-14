"""Smoke test for the new 3-level {4,8,16} SHMQ-Ultimate components.

Tests:
1. Import all new modules
2. ILP 3-level solver on a synthetic problem
3. ISA-aware quanta matching
4. 3-level decoupled permutation
5. MixLLM adapter weight packing (CPU fallback)
6. SHMQConfig 3-level validation
"""
from __future__ import annotations
import sys
import os
import json

# Ensure src/ is on PYTHONPATH
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)  # parent of scripts/
SRC = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC)

import torch
import torch.nn as nn

print("=" * 70)
print("SHMQ-Ultimate 3-Level Smoke Test")
print("=" * 70)

# -------- 1. Imports --------
print("\n[1] Importing new modules...")
from shmq.config import SHMQConfig
from shmq.ilp import solve_ilp_3level, ILPResult3L, compute_target_avg_bits
from shmq.polyq import (
    apply_isa_matching, isa_match_cluster_sizes, cluster_sizes_to_indices,
    TENSOR_CORE_TILE,
)
from shmq.permutation import (
    decoupled_permutation_3level,
    apply_permutation_to_parallel_layers_3level,
)
from shmq.mixllm import (
    SHMQMixLLMLinear, SHMQMixLLMConfig,
    convert_linear_to_mixllm, convert_model_to_mixllm,
    is_mixllm_available, pack_int4_weights, pack_int8_weights,
)
print("  All imports OK.")

# -------- 2. SHMQConfig 3-level --------
print("\n[2] Testing SHMQConfig 3-level...")
cfg = SHMQConfig(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    device="cpu",  # for testing
    target_hp_ratio_16=0.05,
    target_hp_ratio_8=0.20,
    use_3level_ilp=True,
    enable_isa_matching=True,
)
print(cfg.summary())
print(f"  Computed target avg bits: {cfg.computed_target_avg_bits:.3f}")
assert abs(cfg.computed_target_avg_bits - 5.4) < 0.01, \
    f"Expected 5.4 bits, got {cfg.computed_target_avg_bits}"
print("  Config OK.")

# -------- 3. ILP 3-level solver --------
print("\n[3] Testing 3-level ILP solver...")
n_layers = 10
layer_names = [f"layer_{i}" for i in range(n_layers)]
sensitivities = {n: float(torch.rand(1).item() * 10) for n in layer_names}
n_params = {n: 4096 * 4096 for n in layer_names}
qerr_4 = {n: float(torch.rand(1).item() * 100) for n in layer_names}
qerr_8 = {n: q * 0.1 for n, q in qerr_4.items()}  # 8-bit error << 4-bit
qerr_16 = {n: 0.0 for n in layer_names}
parallel_groups = {
    "attn_0": ["layer_0", "layer_1", "layer_2"],
    "ffn_0":  ["layer_3", "layer_4"],
}

result = solve_ilp_3level(
    layer_names=layer_names,
    sensitivities=sensitivities,
    n_params=n_params,
    quant_error_4bit=qerr_4,
    quant_error_8bit=qerr_8,
    quant_error_16bit=qerr_16,
    target_avg_bits=5.4,
    min_avg_bits=4.5,
    parallel_groups=parallel_groups,
    solver="CBC",
    time_limit=10,
    verbose=False,
)
print(result.summary())
assert result.n_layers_4bit + result.n_layers_8bit + result.n_layers_16bit == n_layers
assert result.total_bits <= 5.4 + 1e-3, f"Avg bits {result.total_bits} > budget 5.4"
# Verify parallel constraint
assert result.bit_allocation["layer_0"] == result.bit_allocation["layer_1"] == result.bit_allocation["layer_2"], \
    "Parallel constraint violated for attention group"
assert result.bit_allocation["layer_3"] == result.bit_allocation["layer_4"], \
    "Parallel constraint violated for FFN group"
print("  ILP 3-level OK.")

# -------- 4. ISA-aware quanta matching --------
print("\n[4] Testing ISA-aware quanta matching...")
# Simulate a layer with 4096 output channels and 5%/20%/75% split
n_channels = 4096
ratios = {16: 0.05, 8: 0.20, 4: 0.75}
k16_init = int(n_channels * ratios[16])
k8_init  = int(n_channels * ratios[8])
k4_init  = n_channels - k16_init - k8_init
print(f"  Initial: k16={k16_init}, k8={k8_init}, k4={k4_init} (sum={k16_init+k8_init+k4_init})")

k16, k8, k4 = isa_match_cluster_sizes(
    n_channels=n_channels,
    initial_k16=k16_init, initial_k8=k8_init, initial_k4=k4_init,
    avg_bits_budget=5.4,
    tile_16=TENSOR_CORE_TILE[16], tile_8=TENSOR_CORE_TILE[8], tile_4=TENSOR_CORE_TILE[4],
    prefer_upgrade=True,
)
print(f"  ISA-matched: k16={k16}, k8={k8}, k4={k4} (sum={k16+k8+k4})")
print(f"  Tile alignment: k16%128={k16%128}, k8%128={k8%128}, k4%64={k4%64}")
assert k16 + k8 + k4 == n_channels, "Cluster sizes don't sum to n_channels"
assert k16 % 128 == 0 or k16 == 0, f"k16={k16} not aligned to 128"
assert k8  % 128 == 0 or k8  == 0, f"k8={k8} not aligned to 128"
# k4 may have a partial tile (acceptable)
avg = (16*k16 + 8*k8 + 4*k4) / n_channels
print(f"  Avg bits after ISA matching: {avg:.3f}")
assert avg <= 5.4 + 1e-3, f"Avg bits {avg} exceeds budget 5.4"
print("  ISA matching OK.")

# -------- 5. 3-level decoupled permutation --------
print("\n[5] Testing 3-level decoupled permutation...")
# Use cin=4096 so that 5% ratio gives 204 channels → ISA-rounded to 128 (non-empty C16)
cin = 4096
torch.manual_seed(42)
channel_sens = torch.rand(cin)
perm_metric = torch.rand(cin) * 10

perm, cs = decoupled_permutation_3level(
    channel_sensitivity=channel_sens,
    permutation_metric=perm_metric,
    ratio_16=0.05, ratio_8=0.20,
    tile_16=128, tile_8=128, tile_4=64,
)
print(f"  Cluster sizes: {cs}")
print(f"  Permutation: shape={perm.shape}, dtype={perm.dtype}")
assert perm.numel() == cin
assert len(set(perm.tolist())) == cin, "Permutation has duplicates"
assert cs[16] + cs[8] + cs[4] == cin
# Verify: top of permutation should be highest-sensitivity channels (C16)
# and bottom should be lowest-sensitivity (C4).
if cs[16] > 0:
    top_sens = channel_sens[perm[:cs[16]]]
    bot_sens = channel_sens[perm[cs[16]+cs[8]:]]
    print(f"  Avg sens C16: {top_sens.mean():.3f}  (should be HIGH)")
    print(f"  Avg sens C4:  {bot_sens.mean():.3f}  (should be LOW)")
    assert top_sens.mean() > bot_sens.mean(), "C16 should have higher sensitivity than C4"
else:
    print(f"  C16 is empty (cin too small for tile_16=128); skipping sensitivity check.")
print("  3-level permutation OK.")

# -------- 6. MixLLM adapter (CPU fallback) --------
print("\n[6] Testing MixLLM adapter (CPU fallback)...")
# Create a small linear layer
in_features = 256
out_features = 512
linear = nn.Linear(in_features, out_features, bias=True, dtype=torch.float32)
linear.weight.data = torch.randn(out_features, in_features, dtype=torch.float32) * 0.1
linear.bias.data = torch.randn(out_features, dtype=torch.float32) * 0.01

# Pack weights
w_int8, s_int8 = pack_int8_weights(linear.weight.data.to(torch.float16))
print(f"  INT8 packed: weight={w_int8.shape} {w_int8.dtype}, scale={s_int8.shape} {s_int8.dtype}")
assert w_int8.shape == (out_features, in_features)
assert w_int8.dtype == torch.int8

w_int4, s_int4, z_int4 = pack_int4_weights(linear.weight.data.to(torch.float16))
print(f"  INT4 packed: weight={w_int4.shape} {w_int4.dtype}, scale={s_int4.shape}, zero={z_int4.shape}")
assert w_int4.shape == (out_features, in_features // 2)
assert w_int4.dtype == torch.uint8

# Convert to SHMQMixLLMLinear
perm = torch.randperm(out_features)
cs = {16: 64, 8: 128, 4: out_features - 64 - 128}  # 64+128+320=512
new_module = convert_linear_to_mixllm(
    linear,
    bit_allocation_for_layer=4,
    permutation_indices=perm,
    cluster_sizes=cs,
    group_size=128,
)
print(f"  Converted module: {new_module}")
print(f"    n_fp16={new_module.n_fp16}, n_int8={new_module.n_int8}, n_int4={new_module.n_int4}")
assert new_module.n_fp16 == cs[16]
assert new_module.n_int8 == cs[8]
assert new_module.n_int4 == cs[4]

# Forward pass (CPU fallback)
x = torch.randn(8, in_features, dtype=torch.float32)
y = new_module(x)
print(f"  Forward output: shape={y.shape}, dtype={y.dtype}")
assert y.shape == (8, out_features)
assert torch.isfinite(y).all(), "Output has NaN/Inf"
print("  MixLLM adapter (CPU fallback) OK.")

# -------- 7. Conversion summary --------
print("\n[7] MixLLM available:", is_mixllm_available())
print("\n" + "=" * 70)
print("ALL SMOKE TESTS PASSED")
print("=" * 70)
