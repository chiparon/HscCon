"""Deterministic assertions for the v2 Attention candidate."""

from __future__ import annotations

import importlib.util

import torch

from tools.hif4_benchmark import validate_hif4_params
from workbench.benchmark_v2_attention_synthetic import build_suite


def _load():
    spec = importlib.util.spec_from_file_location(
        "v2_attention_candidate", "workbench/solution_v2_attention_candidate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    module = _load()
    generator = torch.Generator().manual_seed(913)
    for q_heads, kv_heads, head_dim in ((4, 4, 64), (8, 2, 128), (16, 2, 256)):
        q = torch.randn((11, q_heads * head_dim), generator=generator)
        k = torch.randn((13, kv_heads * head_dim), generator=generator)
        smooth_k = torch.exp(
            0.2 * torch.randn((kv_heads, head_dim), generator=generator)
        )
        smooth_q = smooth_k.repeat_interleave(q_heads // kv_heads, dim=0)
        qt = module._transform_qk(q, smooth_q, q_heads, head_dim, True)
        kt = module._transform_qk(k, smooth_k, kv_heads, head_dim, False)
        qh = q.reshape(11, q_heads, head_dim).transpose(0, 1)
        kh = k.reshape(13, kv_heads, head_dim).transpose(0, 1)
        qth = qt.reshape(11, q_heads, head_dim).transpose(0, 1)
        kth = kt.reshape(13, kv_heads, head_dim).transpose(0, 1)
        repeat = q_heads // kv_heads
        kh = kh.repeat_interleave(repeat, dim=0)
        kth = kth.repeat_interleave(repeat, dim=0)
        reference = qh @ kh.transpose(-1, -2)
        transformed = qth @ kth.transpose(-1, -2)
        tolerance = 2.0e-5 * max(1.0, float(reference.abs().max()))
        assert float((reference - transformed).abs().max()) <= tolerance

    for group in build_suite():
        q_heads = group["q_num_heads"]
        kv_heads = group["kv_num_heads"]
        head_dim = group["head_dim"]
        state = module.hif4_calibration_attention(
            group["calib"], q_heads, kv_heads, head_dim
        )
        assert state["q_state"]["smooth"].shape == (q_heads, head_dim)
        assert state["k_state"]["smooth"].shape == (kv_heads, head_dim)
        assert state["q_state"]["sensitivity"].shape == (q_heads, head_dim)
        assert state["k_state"]["sensitivity"].shape == (kv_heads, head_dim)
        for sample in group["test"]:
            qp = module.hif4_dynamic_quantize_q(
                *sample["q"], q_heads, head_dim, state["q_state"]
            )
            kp = module.hif4_dynamic_quantize_k(
                *sample["k"], kv_heads, head_dim, state["k_state"]
            )
            vp = module.hif4_dynamic_quantize_v(
                *sample["v"], kv_heads, head_dim, state["v_state"]
            )
            validate_hif4_params(qp, sample["q"][0].shape)
            validate_hif4_params(kp, sample["k"][0].shape)
            validate_hif4_params(vp, sample["v"][0].shape)
    print("v2 attention candidate: all assertions passed")


if __name__ == "__main__":
    main()
