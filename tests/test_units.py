"""Unit smoke tests for SHMQ-Ultimate core components (CPU, no model)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from shmq_ultimate.quant_utils import (fake_quantize_group_sym, pack_int4,
                                       quantize_activation_per_token,
                                       quantize_group_sym, unpack_int4)
from shmq_ultimate.sensitivity import obs_inverse_diag, intra_channel_sensitivity
from shmq_ultimate.permutation import (ChannelPartition, decoupled_permutation,
                                       partition_sizes, round_to_quanta)
from shmq_ultimate.rmsnorm_fusion import PermutedRMSNorm
from shmq_ultimate.autoround import autoround_layer
from shmq_ultimate.sqc import sqc_calibrate_scales
from shmq_ultimate.gptq import gptq_quantize
from shmq_ultimate.layer_quantizer import bits_vector, quantize_layer
from shmq_ultimate.inference.quant_linear import SHMQUltimateLinear

torch.manual_seed(0)
PASS = []


def check(name, cond):
    assert cond, f"FAIL: {name}"
    PASS.append(name)
    print(f"  ok  {name}")


def test_quant_utils():
    w = torch.randn(64, 256)
    codes, scales = quantize_group_sym(w, 4, 128)
    check("int4 codes in range", codes.min() >= -8 and codes.max() <= 7)
    wq = fake_quantize_group_sym(w, 8, 128)
    check("int8 fakequant close", (w - wq).abs().max() < 0.05)
    p = pack_int4(codes)
    check("pack/unpack roundtrip", torch.equal(unpack_int4(p).to(torch.int16), codes))
    x = torch.randn(7, 256) * 3
    qx, sx = quantize_activation_per_token(x, 8)
    err = (qx.float() * sx - x).abs().max()
    check("per-token act quant", err < 0.2)


def test_sensitivity():
    X = torch.randn(512, 128)
    H = X.t() @ X / 512
    d = obs_inverse_diag(H, 0.1)
    check("obs inv diag positive", bool((d > 0).all()))
    w = torch.randn(64, 128)
    s = intra_channel_sensitivity(w, H, 4, 128)
    check("channel sens shape", s.shape == (128,) and bool((s >= 0).all()))


def test_permutation():
    check("round_to_quanta", round_to_quanta(300, 128, 1024) == 256)
    n16, n8, n4 = partition_sizes(1024, 4, 0.125, 128, 64)
    check("partition sizes int4", n16 == 0 and n8 == 128 and n4 == 896
          and n4 % 64 == 0)
    n16, n8, n4 = partition_sizes(1024, 8, 0.125, 128, 64)
    check("partition sizes int8", n16 == 128 and n8 == 896 and n4 == 0)
    sens = torch.rand(1024); mag = torch.rand(1024)
    perm = decoupled_permutation(sens, mag, 0, 128, 896)
    check("perm is permutation", torch.equal(perm.sort().values, torch.arange(1024)))
    top = perm[:128]
    check("top cluster most sensitive",
          sens[top].min() >= sens[perm[128:]].max())


def test_rmsnorm_fusion():
    dim = 64
    gamma = torch.rand(dim) + 0.5
    perm = torch.randperm(dim)
    x = torch.randn(3, 5, dim)
    # reference: rmsnorm then gather
    var = x.pow(2).mean(-1, keepdim=True)
    ref = (x * torch.rsqrt(var + 1e-6) * gamma)[..., perm]
    mod = PermutedRMSNorm(gamma, 1e-6, perm)
    out = mod(x)
    check("PermutedRMSNorm == gather(rmsnorm)", (out - ref).abs().max() < 1e-5)


def test_autoround_sqc_gptq():
    torch.manual_seed(1)
    w = torch.randn(32, 256)
    X = torch.randn(64, 256)
    H = X.t() @ X / 64
    # AutoRound improves output MSE
    from shmq_ultimate.quant_utils import group_scales, qmax
    s = group_scales(w, 4, 128)
    V = autoround_layer(w, 4, 128, [X], iters=50, lr=1e-2, scales=s)
    check("V bounded", bool((V.abs() <= 0.5 + 1e-6).all()))
    def out_err(V_):
        wg = w.view(32, 2, 128); sv = s.unsqueeze(-1)
        q = torch.clamp(torch.round(wg / sv + V_.view(32, 2, 128)), -8, 7) * sv
        return float(((X @ (q.view(32, 256) - w).t()) ** 2).mean())
    check("autoround reduces output err", out_err(V) <= out_err(torch.zeros_like(V)) + 1e-9)
    # SQC returns valid scales
    sal = torch.diagonal(H)
    s2 = sqc_calibrate_scales(w, 4, 128, sal, grid=10, search_range=0.1)
    check("sqc scales positive", bool((s2 > 0).all()))
    # mixed-bit GPTQ
    part = ChannelPartition(perm=torch.arange(256), n16=0, n8=128, n4=128)
    bpc = bits_vector(256, part)
    Q, codes, scales_used = gptq_quantize(w, H, bpc, 128)
    check("gptq int8 codes range",
          codes[:, :128].abs().max() <= 128 and codes[:, 128:].abs().max() <= 8)
    rtn = fake_quantize_group_sym(w, 4, 128)
    e_gptq = float(((X @ (Q - w).t()) ** 2).mean())
    e_rtn_mixed = None
    check("gptq finite err", e_gptq == e_gptq)  # not NaN


def test_quantize_layer_and_linear():
    torch.manual_seed(2)
    cout, cin, g = 32, 256, 128
    w = torch.randn(cout, cin)
    X = torch.randn(64, cin)
    H = X.t() @ X / 64
    part = ChannelPartition(perm=torch.arange(cin), n16=0, n8=128, n4=128)
    res = quantize_layer(w, H, part, g, [X], torch.diagonal(H),
                         autoround_iters=30)
    check("layer result shapes", res.w_deq.shape == (cout, cin)
          and res.codes.shape == (cout, cin))
    lin = SHMQUltimateLinear.from_quantized(res, g)
    wd = lin.dequantize_weight()
    check("packed dequant == gptq dequant", (wd - res.w_deq).abs().max() < 1e-4)
    y_ref = X @ res.w_deq.t()
    y = lin(X)
    rel = ((y - y_ref).norm() / y_ref.norm()).item()
    check(f"forward close to dequant matmul (rel={rel:.4f})", rel < 0.05)
    mem_fp16 = cout * cin * 2
    check("memory smaller than fp16", lin.memory_bytes() < mem_fp16)


if __name__ == "__main__":
    for fn in [test_quant_utils, test_sensitivity, test_permutation,
               test_rmsnorm_fusion, test_autoround_sqc_gptq,
               test_quantize_layer_and_linear]:
        print(f"[{fn.__name__}]")
        fn()
    print(f"\nALL {len(PASS)} UNIT CHECKS PASSED")
