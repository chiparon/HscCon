"""Five-base ablation for the guarded multibase V2 candidate."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workbench import solution_v2_multibase_candidate as _impl


@_impl.torch.inference_mode()
def hif4_calibration_and_quantize_weight(weight_quant, weight_scale, calib):
    del calib
    return {
        "weight_params": _impl._quantize_hif4_five(weight_quant, weight_scale),
        "activation_state": {"version": 3, "base_search": 5},
    }


@_impl.torch.inference_mode()
def hif4_dynamic_quantize_activation(activation_quant, activation_scale, state):
    del state
    return _impl._quantize_hif4_five(activation_quant, activation_scale)


def hif4_calibration_attention(calib, q_num_heads, kv_num_heads, head_dim):
    return _impl.hif4_calibration_attention(
        calib, q_num_heads, kv_num_heads, head_dim
    )


def hif4_dynamic_quantize_q(q, scale, heads, head_dim, state):
    return _impl.hif4_dynamic_quantize_q(q, scale, heads, head_dim, state)


def hif4_dynamic_quantize_k(k, scale, heads, head_dim, state):
    return _impl.hif4_dynamic_quantize_k(k, scale, heads, head_dim, state)


@_impl.torch.inference_mode()
def hif4_dynamic_quantize_v(v, scale, heads, head_dim, state):
    del heads, head_dim, state
    return _impl._quantize_hif4_five(v, scale)
