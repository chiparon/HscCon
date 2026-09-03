"""Slow, exhaustive HiF4 reference used only for local performance comparisons.

The implementation deliberately enumerates all eight legal level-2/level-3
layouts for each eight-value group.  ``solution.py`` is expected to produce the
same result with an algebraically reduced hierarchy solver.
"""

from __future__ import annotations

from typing import Any

import torch


_E6M2_MIN = 2.0**-48
_E6M2_MAX = 49152.0


def _e6m2_nearest(value: torch.Tensor) -> torch.Tensor:
    x = value.to(torch.float32).clamp(_E6M2_MIN, _E6M2_MAX)
    fraction, exponent = torch.frexp(x)
    normalized = fraction * 2.0
    exponent = exponent - 1
    mantissa = torch.round((normalized - 1.0) * 4.0).to(torch.int32)
    carry = mantissa == 4
    exponent = exponent + carry.to(exponent.dtype)
    mantissa = torch.where(carry, torch.zeros_like(mantissa), mantissa)
    exponent = exponent.clamp(-48, 15)
    mantissa = mantissa.clamp(0, 3)
    return torch.ldexp(1.0 + mantissa.to(torch.float32) * 0.25, exponent)


def _encode(dense: torch.Tensor) -> dict[str, torch.Tensor]:
    prefix = dense.shape[:-1]
    channels = int(dense.shape[-1])
    if channels % 64:
        raise ValueError("last dimension must be divisible by 64")
    blocks = channels // 64
    grouped = dense.to(torch.float32).reshape(*prefix, blocks, 8, 2, 4)
    absolute = grouped.abs()
    sign = torch.sign(grouped)
    amax = absolute.amax(dim=(-1, -2, -3))
    raw_scale = (amax.to(torch.bfloat16) * 0.142578125).to(torch.float32)
    scale = _e6m2_nearest(raw_scale)

    best_loss = None
    best_l2 = None
    best_l3a = None
    best_l3b = None
    best_mant = None
    for l2, l3a, l3b in (
        (1, 1, 1),
        (1, 1, 2),
        (1, 2, 1),
        (1, 2, 2),
        (2, 1, 1),
        (2, 1, 2),
        (2, 2, 1),
        (2, 2, 2),
    ):
        l3 = torch.tensor((l3a, l3b), dtype=torch.float32, device=dense.device)
        denominator = scale[..., None, None, None] * float(l2) * l3[..., None]
        mant = torch.round(absolute * (4.0 / denominator)).clamp(0, 7) * 0.25
        reconstructed = mant * denominator
        loss = (absolute - reconstructed).square().sum(dim=(-1, -2))
        if best_loss is None:
            take = torch.ones_like(loss, dtype=torch.bool)
            best_loss = loss
            best_l2 = torch.full_like(loss, l2, dtype=torch.float32)
            best_l3a = torch.full_like(loss, l3a, dtype=torch.float32)
            best_l3b = torch.full_like(loss, l3b, dtype=torch.float32)
            best_mant = mant
        else:
            take = loss < best_loss
            best_loss = torch.where(take, loss, best_loss)
            best_l2 = torch.where(take, float(l2), best_l2)
            best_l3a = torch.where(take, float(l3a), best_l3a)
            best_l3b = torch.where(take, float(l3b), best_l3b)
            best_mant = torch.where(take[..., None, None], mant, best_mant)

    assert best_mant is not None
    scale_lv3 = torch.stack((best_l3a, best_l3b), dim=-1)
    sign = torch.where(best_mant == 0, torch.zeros_like(sign), sign)
    return {
        "scale_factor": scale.reshape(*prefix, blocks, 1, 1, 1),
        "scale_lv2": best_l2.reshape(*prefix, blocks, 8, 1, 1),
        "scale_lv3": scale_lv3.reshape(*prefix, blocks, 8, 2, 1),
        "sign": sign,
        "mant": best_mant,
    }


def _nvfp4(quant: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (quant.unflatten(-1, (-1, 16)) * scale.unsqueeze(-1)).flatten(-2).to(torch.bfloat16)


def hif4_calibration_and_quantize_weight(weight_quant, weight_scale, calib_activation_list):
    return {"weight_params": _encode(_nvfp4(weight_quant, weight_scale)), "activation_state": None}


def hif4_dynamic_quantize_activation(activation_quant, activation_scale, activation_state):
    return _encode(_nvfp4(activation_quant, activation_scale))


def hif4_calibration_attention(calib_qkv_list, q_num_heads, kv_num_heads, head_dim):
    return {"q_state": None, "k_state": None, "v_state": None}


def hif4_dynamic_quantize_q(q_quant, q_scale, q_num_heads, head_dim, q_state):
    return _encode(_nvfp4(q_quant, q_scale))


def hif4_dynamic_quantize_k(k_quant, k_scale, kv_num_heads, head_dim, k_state):
    return _encode(_nvfp4(k_quant, k_scale))


def hif4_dynamic_quantize_v(v_quant, v_scale, kv_num_heads, head_dim, v_state):
    return _encode(_nvfp4(v_quant, v_scale))
