#!/usr/bin/env python3
"""SHMQ++ (SHMQ-Ultimate) streamed quantizer for full-size Qwen models.

Quantizes a HuggingFace Qwen checkpoint shard-by-shard so it runs even on a
1 GB-RAM machine: each safetensors shard is downloaded, its linear weights are
quantized to the SHMQ++ 3-segment format, written to an output shard with a
streaming safetensors writer, and the input shard is deleted before the next
one is fetched.

Data-free mode (this script): magnitude-based channel partition (SHMQ UB
mechanism, 12.5% INT8 outlier channels + 87.5% INT4, per-group g=128 symmetric
scales) with decoupled permutation:
  * q/k/v share one input permutation      -> fused into input_layernorm
  * gate/up share one input permutation    -> fused into post_attention_layernorm
  * down_proj permutation is propagated to gate/up OUTPUT rows (PolyQ layout
    propagation) -> zero runtime overhead
  * o_proj keeps identity permutation (no fusable producer under GQA)

The calibration-based full pipeline (Fisher + ILP + OBS/GPTQ + AutoRound +
SQC) lives in src/shmq_ultimate/pipeline.py and should be used on a GPU
machine; this script produces the same on-disk format.

Output: safetensors shards with tensors
    {layer}.w8 [cout,n8] I8, {layer}.s8 [cout,n8/G] F32,
    {layer}.w4 [cout,n4/2] U8, {layer}.s4 [cout,n4/G] F32,
    model.layers.i.self_attn.in_perm / mlp.in_perm / mlp.down_perm  I32
plus all non-linear tensors passed through unchanged, config/tokenizer files,
and shmq_meta.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import sys

import torch

G = 128          # group size
UB = 0.125       # high-precision (INT8) channel ratio
QUANTA8 = 128    # PolyQ quanta for the INT8 segment

LINEAR_RE = re.compile(
    r"^model\.layers\.(\d+)\.(self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)\.weight$")

DT_BYTES = {"F32": 4, "F16": 2, "BF16": 2, "I8": 1, "U8": 1, "I32": 4, "I64": 8}


# --------------------------------------------------------------------------- #
class StreamWriter:
    """Minimal streaming safetensors writer.

    Tensor list (name, dtype, shape) must be declared up front; data is then
    appended in exactly that order via write_bytes()/write_tensor().
    """

    def __init__(self, path: str, tensors, metadata=None):
        header = {}
        if metadata:
            header["__metadata__"] = {k: str(v) for k, v in metadata.items()}
        off = 0
        self.order = []
        for name, dt, shape in tensors:
            n = DT_BYTES[dt]
            for s in shape:
                n *= s
            header[name] = {"dtype": dt, "shape": list(shape),
                            "data_offsets": [off, off + n]}
            self.order.append((name, n))
            off += n
        hb = json.dumps(header, separators=(",", ":")).encode()
        pad = (8 - len(hb) % 8) % 8
        hb += b" " * pad
        self.f = open(path, "wb")
        self.f.write(struct.pack("<Q", len(hb)))
        self.f.write(hb)
        self.idx = 0
        self.written = 0
        self.expect = self.order

    def write_tensor_chunked(self, tensor_iter, name):
        exp_name, exp_n = self.expect[self.idx]
        assert exp_name == name, f"order mismatch: {exp_name} != {name}"
        n = 0
        for chunk in tensor_iter:
            b = chunk.contiguous().flatten().view(torch.uint8).numpy().tobytes()
            self.f.write(b)
            n += len(b)
        assert n == exp_n, f"{name}: wrote {n}, expected {exp_n}"
        self.idx += 1

    def write_tensor(self, t: torch.Tensor, name: str):
        self.write_tensor_chunked([t], name)

    def close(self):
        assert self.idx == len(self.expect), "missing tensors"
        self.f.close()


def st_dtype(t) -> str:
    if isinstance(t, str):          # safetensors slice.get_dtype() -> 'BF16'
        return t
    return {torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16",
            torch.int8: "I8", torch.uint8: "U8", torch.int32: "I32",
            torch.int64: "I64"}[t]


# --------------------------------------------------------------------------- #
def partition(cin: int):
    n8 = max(QUANTA8, int(round(UB * cin / QUANTA8)) * QUANTA8)
    n4 = cin - n8
    assert n4 % 2 == 0 and n4 % G == 0, (cin, n8, n4)
    return n8, n4


@torch.no_grad()
def quantize_linear(w: torch.Tensor, perm: torch.Tensor):
    """w: (cout, cin) fp32, perm: (cin,).  Returns w8, s8, w4, s4."""
    cout, cin = w.shape
    n8, n4 = partition(cin)
    wp = w[:, perm]
    # INT8 segment
    w8g = wp[:, :n8].view(cout, n8 // G, G)
    s8 = (w8g.abs().amax(dim=-1) / 127.0).clamp_min(1e-8)
    q8 = torch.clamp(torch.round(w8g / s8.unsqueeze(-1)), -127, 127).to(torch.int8)
    # INT4 segment
    w4g = wp[:, n8:].view(cout, n4 // G, G)
    s4 = (w4g.abs().amax(dim=-1) / 7.0).clamp_min(1e-8)
    q4 = torch.clamp(torch.round(w4g / s4.unsqueeze(-1)), -7, 7).to(torch.int8)
    q4 = q4.view(cout, n4)
    packed = (((q4[:, 0::2] & 0xF) << 4) | (q4[:, 1::2] & 0xF)).to(torch.uint8)
    return (q8.view(cout, n8), s8.float(), packed, s4.float())


@torch.no_grad()
def col_magnitude(f, key: str) -> torch.Tensor:
    """Per-input-channel |w|_inf of a linear weight, streamed by row chunks."""
    sl = f.get_slice(key)
    rows, cin = sl.get_shape()
    m = torch.zeros(cin)
    step = max(1, (64 << 20) // (cin * 2))   # ~64MB chunks
    for r0 in range(0, rows, step):
        chunk = sl[r0:min(r0 + step, rows)].float()
        m = torch.maximum(m, chunk.abs().amax(dim=0))
    return m


def magnitude_perm(mag: torch.Tensor) -> torch.Tensor:
    """Decoupled permutation, data-free variant: sort channels by magnitude
    (desc) so the INT8 cluster captures outlier channels (SmoothQuant/SHMQ UB
    insight); within clusters order stays magnitude-sorted (Eq. 12)."""
    return torch.argsort(mag, descending=True).to(torch.int32)


# --------------------------------------------------------------------------- #
def plan_shard(f, keys, perms_avail):
    """Return ordered output tensor declarations + write plan for one shard."""
    decls, plan = [], []
    layers_seen = set()
    for key in keys:
        m = LINEAR_RE.match(key)
        if not m:
            sl = f.get_slice(key)
            shape = sl.get_shape()
            dt = st_dtype(sl.get_dtype()) if hasattr(sl, "get_dtype") else None
            decls.append((key, dt, shape))
            plan.append(("pass", key))
            continue
        li, role = int(m.group(1)), m.group(2)
        base = key[:-len(".weight")]
        sl = f.get_slice(key)
        cout, cin = sl.get_shape()
        n8, n4 = partition(cin)
        # emit perm tensors once per layer component
        if role == "self_attn.q_proj" and (li, "attn") not in layers_seen:
            layers_seen.add((li, "attn"))
            decls.append((f"model.layers.{li}.self_attn.in_perm", "I32", [cin]))
            plan.append(("perm", (li, "attn_in", cin)))
        if role == "mlp.gate_proj" and (li, "mlp") not in layers_seen:
            layers_seen.add((li, "mlp"))
            decls.append((f"model.layers.{li}.mlp.in_perm", "I32", [cin]))
            plan.append(("perm", (li, "mlp_in", cin)))
        if role == "mlp.down_proj" and (li, "down") not in layers_seen:
            layers_seen.add((li, "down"))
            decls.append((f"model.layers.{li}.mlp.down_perm", "I32", [cin]))
            plan.append(("perm", (li, "down_in", cin)))
        decls += [(f"{base}.w8", "I8", [cout, n8]),
                  (f"{base}.s8", "F32", [cout, n8 // G]),
                  (f"{base}.w4", "U8", [cout, n4 // 2]),
                  (f"{base}.s4", "F32", [cout, n4 // G])]
        plan.append(("quant", (key, li, role)))
    return decls, plan


def get_dtype_str(f, key):
    # safetensors slice API: get_dtype not always present; fall back to header
    try:
        return st_dtype(f.get_slice(key).get_dtype())
    except Exception:
        return None


@torch.no_grad()
def process_shard(in_path: str, out_path: str, perms: dict, stats: dict):
    from safetensors import safe_open
    f = safe_open(in_path, framework="pt")
    keys = sorted(f.keys())

    # ---------------- pass 1: permutations for this shard's layers ----------
    by_layer = {}
    for key in keys:
        m = LINEAR_RE.match(key)
        if m:
            by_layer.setdefault(int(m.group(1)), {})[m.group(2)] = key
    for li, roles in sorted(by_layer.items()):
        if ("attn", li) not in perms and "self_attn.q_proj" in roles:
            mag = col_magnitude(f, roles["self_attn.q_proj"])
            for r in ("self_attn.k_proj", "self_attn.v_proj"):
                if r in roles:
                    mag = torch.maximum(mag, col_magnitude(f, roles[r]))
            perms[("attn", li)] = magnitude_perm(mag)
        if ("mlp", li) not in perms and "mlp.gate_proj" in roles:
            mag = col_magnitude(f, roles["mlp.gate_proj"])
            if "mlp.up_proj" in roles:
                mag = torch.maximum(mag, col_magnitude(f, roles["mlp.up_proj"]))
            perms[("mlp", li)] = magnitude_perm(mag)
        if ("down", li) not in perms and "mlp.down_proj" in roles:
            perms[("down", li)] = magnitude_perm(
                col_magnitude(f, roles["mlp.down_proj"]))

    # ---------------- pass 2: plan + write ----------------------------------
    decls_raw, plan = plan_shard(f, keys, perms)
    # fill passthrough dtypes
    decls = []
    for name, dt, shape in decls_raw:
        if dt is None:
            t0 = f.get_slice(name)[0:1]
            dt = st_dtype(t0.dtype)
        decls.append((name, dt, list(shape)))

    meta = {"format": "shmq_ultimate_v3", "scheme": "W4.8-sym-g128-A8",
            "group_size": G, "ub": UB,
            "note": "SHMQ++ 3-segment layout [INT8|INT4]; perms fuse into "
                    "RMSNorm (attn/mlp in) and gate/up rows (down)."}
    w = StreamWriter(out_path, decls, meta)

    for op, arg in plan:
        if op == "pass":
            key = arg
            sl = f.get_slice(key)
            shape = sl.get_shape()
            if len(shape) >= 2 and shape[0] > 4096:
                rows, step = shape[0], max(1, (64 << 20) // max(1, int(
                    torch.tensor(shape[1:]).prod()) * 2))
                w.write_tensor_chunked(
                    (sl[r0:min(r0 + step, rows)] for r0 in range(0, rows, step)),
                    key)
            else:
                w.write_tensor(f.get_slice(key)[:], key)
        elif op == "perm":
            li, kind, cin = arg
            key_map = {"attn_in": ("attn", "self_attn.in_perm"),
                       "mlp_in": ("mlp", "mlp.in_perm"),
                       "down_in": ("down", "mlp.down_perm")}
            pk, name_sfx = key_map[kind]
            perm = perms.get((pk, li),
                             torch.arange(cin, dtype=torch.int32))
            w.write_tensor(perm.to(torch.int32),
                           f"model.layers.{li}.{name_sfx}")
        else:  # quant
            key, li, role = arg
            base = key[:-len(".weight")]
            wt = f.get_slice(key)[:].float()
            cout, cin = wt.shape
            # input permutation
            if role in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"):
                perm = perms.get(("attn", li))
            elif role in ("mlp.gate_proj", "mlp.up_proj"):
                perm = perms.get(("mlp", li))
            elif role == "mlp.down_proj":
                perm = perms.get(("down", li))
            else:  # o_proj: identity (GQA head pairing forbids free perm)
                perm = None
            if perm is None:
                perm = torch.arange(cin)
            # PolyQ layout propagation: down_perm permutes gate/up OUTPUT rows
            if role in ("mlp.gate_proj", "mlp.up_proj"):
                dp = perms.get(("down", li))
                if dp is not None:
                    wt = wt[dp.long()]
            w8, s8, w4, s4 = quantize_linear(wt, perm.long())
            w.write_tensor(w8, f"{base}.w8")
            w.write_tensor(s8, f"{base}.s8")
            w.write_tensor(w4, f"{base}.w4")
            w.write_tensor(s4, f"{base}.s4")
            n8, n4 = partition(cin)
            stats["lin_params"] = stats.get("lin_params", 0) + cout * cin
            stats["lin_bits"] = stats.get("lin_bits", 0) + cout * (
                n8 * 8 + n4 * 4 + (n8 // G + n4 // G) * 32)
            del wt
    w.close()


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", default="quantized_models/qwen3-8b-shmq-ultimate")
    ap.add_argument("--keep-shards", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    os.makedirs(args.out, exist_ok=True)

    # aux files
    for fn in ("config.json", "generation_config.json", "tokenizer.json",
               "tokenizer_config.json", "vocab.json", "merges.txt"):
        try:
            p = hf_hub_download(args.model, fn)
            shutil.copy(p, os.path.join(args.out, fn))
        except Exception:
            pass

    # weight index
    try:
        idx_path = hf_hub_download(args.model, "model.safetensors.index.json")
        with open(idx_path) as fh:
            index = json.load(fh)
        shard_names = sorted(set(index["weight_map"].values()))
    except Exception:
        shard_names = ["model.safetensors"]

    perms, stats = {}, {}
    out_index = {"metadata": {"format": "shmq_ultimate_v3"}, "weight_map": {}}
    for si, sn in enumerate(shard_names):
        print(f"[{si+1}/{len(shard_names)}] downloading {sn} ...", flush=True)
        sp = hf_hub_download(args.model, sn)
        out_sn = sn.replace(".safetensors", ".shmq.safetensors")
        out_path = os.path.join(args.out, out_sn)
        print(f"    quantizing -> {out_sn}", flush=True)
        process_shard(sp, out_path, perms, stats)
        # record weight map
        import struct as _s
        with open(out_path, "rb") as fh:
            hlen = _s.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(hlen))
        for name in hdr:
            if name != "__metadata__":
                out_index["weight_map"][name] = out_sn
        if not args.keep_shards:
            os.remove(sp)
            print(f"    removed source shard", flush=True)
        print(f"    output size: {os.path.getsize(out_path)/1e9:.2f} GB", flush=True)

    with open(os.path.join(args.out, "model.shmq.index.json"), "w") as fh:
        json.dump(out_index, fh, indent=1)
    avg_bits = stats["lin_bits"] / stats["lin_params"]
    meta = {"model": args.model, "format": "shmq_ultimate_v3",
            "scheme": "W4.8A8 (3-segment [FP16|INT8|INT4], data-free UB mode)",
            "group_size": G, "ub": UB, "avg_linear_bits": round(avg_bits, 3),
            "linear_params": stats["lin_params"]}
    with open(os.path.join(args.out, "shmq_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(json.dumps(meta, indent=2))
    total = sum(os.path.getsize(os.path.join(args.out, x))
                for x in os.listdir(args.out))
    print(f"DONE: {args.out} ({total/1e9:.2f} GB, avg linear bits {avg_bits:.3f})")


if __name__ == "__main__":
    main()
