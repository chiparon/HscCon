"""Attention-only v2 quality/timing benchmark on the public mini sample.

Reports both full and causal scaled-dot-product attention MSE.  Calibration
data alone is passed to candidate state construction; test data is only used
after state has been frozen.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import time
from pathlib import Path

import torch

from tools.hif4_benchmark import (
    NVFP4Tensor,
    dequantize_hif4,
    dequantize_nvfp4,
    validate_hif4_params,
)


def _load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dequant(pair) -> torch.Tensor:
    return dequantize_nvfp4(NVFP4Tensor(pair[0], pair[1]))


def _attention(q, k, v, q_heads, kv_heads, head_dim, causal):
    q = q.reshape(-1, q_heads, head_dim).transpose(0, 1)
    k = k.reshape(-1, kv_heads, head_dim).transpose(0, 1)
    v = v.reshape(-1, kv_heads, head_dim).transpose(0, 1)
    if q_heads != kv_heads:
        repeat = q_heads // kv_heads
        k = k.repeat_interleave(repeat, dim=0)
        v = v.repeat_interleave(repeat, dim=0)
    logits = (q @ k.transpose(-1, -2)) / math.sqrt(head_dim)
    if causal:
        q_len, k_len = logits.shape[-2:]
        mask = torch.ones((q_len, k_len), dtype=torch.bool).triu(1)
        logits = logits.masked_fill(mask, float("-inf"))
    return torch.softmax(logits, dim=-1) @ v


def _params(module, sample, states, q_heads, kv_heads, head_dim):
    qp = module.hif4_dynamic_quantize_q(
        *sample["q"], q_heads, head_dim, states["q_state"]
    )
    kp = module.hif4_dynamic_quantize_k(
        *sample["k"], kv_heads, head_dim, states["k_state"]
    )
    vp = module.hif4_dynamic_quantize_v(
        *sample["v"], kv_heads, head_dim, states["v_state"]
    )
    validate_hif4_params(qp, sample["q"][0].shape)
    validate_hif4_params(kp, sample["k"][0].shape)
    validate_hif4_params(vp, sample["v"][0].shape)
    return qp, kp, vp


def _score(module, group):
    q_heads = group["q_num_heads"]
    kv_heads = group["kv_num_heads"]
    head_dim = group["head_dim"]
    start = time.perf_counter()
    states = module.hif4_calibration_attention(
        group["calib"], q_heads, kv_heads, head_dim
    )
    calibration_seconds = time.perf_counter() - start
    errors = {"full": [], "causal": []}
    dynamic_seconds = []
    for sample in group["test"]:
        start = time.perf_counter()
        qp, kp, vp = _params(
            module, sample, states, q_heads, kv_heads, head_dim
        )
        dynamic_seconds.append(time.perf_counter() - start)
        q, k, v = (_dequant(sample[name]) for name in ("q", "k", "v"))
        qh = dequantize_hif4(qp, q.shape)
        kh = dequantize_hif4(kp, k.shape)
        vh = dequantize_hif4(vp, v.shape)
        for mode, causal in (("full", False), ("causal", True)):
            reference = _attention(q, k, v, q_heads, kv_heads, head_dim, causal)
            actual = _attention(qh, kh, vh, q_heads, kv_heads, head_dim, causal)
            errors[mode].append(float((reference - actual).square().mean()))
    return {
        "calibration_seconds": calibration_seconds,
        "dynamic_seconds": dynamic_seconds,
        "full_mse": errors["full"],
        "causal_mse": errors["causal"],
        "mean_full_mse": statistics.fmean(errors["full"]),
        "mean_causal_mse": statistics.fmean(errors["causal"]),
    }


def _apply_ablation(module, ablation):
    if ablation == "none":
        return
    original = module.hif4_calibration_attention

    def calibration(data, q_heads, kv_heads, head_dim):
        state = original(data, q_heads, kv_heads, head_dim)
        if ablation == "hadamard-only":
            state["q_state"]["smooth"] = torch.ones_like(state["q_state"]["smooth"])
            state["k_state"]["smooth"] = torch.ones_like(state["k_state"]["smooth"])
        elif ablation == "smooth-only":
            for role in ("q_state", "k_state"):
                state[role]["sensitivity"] = None
                state[role]["search_radius"] = 0
        elif ablation == "no-weight":
            for role in ("q_state", "k_state"):
                state[role]["sensitivity"] = None
        return state

    module.hif4_calibration_attention = calibration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="solution.py")
    parser.add_argument(
        "--candidate", default="workbench/solution_v2_attention_candidate.py"
    )
    parser.add_argument(
        "--data", default=".tmp/quantizer-public/data/attn.pt"
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--ablation",
        choices=("none", "hadamard-only", "smooth-only", "no-weight"),
        default="none",
    )
    parser.add_argument("--json", default="workbench/v2_attention_public.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    group = torch.load(args.data, map_location="cpu", weights_only=False)[0]
    baseline = _score(_load(args.baseline, "attention_v1"), group)
    candidate_module = _load(args.candidate, "attention_v2")
    _apply_ablation(candidate_module, args.ablation)
    candidate = _score(candidate_module, group)
    report = {"baseline": baseline, "candidate": candidate}
    for mode in ("full", "causal"):
        old = baseline[f"mean_{mode}_mse"]
        new = candidate[f"mean_{mode}_mse"]
        report[f"{mode}_improvement_percent"] = 100.0 * (old - new) / old
    Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
