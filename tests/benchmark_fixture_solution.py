"""Small valid submission used only to self-test tools/hif4_benchmark.py."""

from __future__ import annotations

from typing import Any

import torch


def _dequantize(quant: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (
        quant.to(torch.float32)
        .unflatten(-1, (-1, 16))
        .mul(scale.to(torch.float32).unsqueeze(-1))
        .flatten(-2, -1)
    )


def _quantize(quant: torch.Tensor, scale: torch.Tensor) -> dict[str, torch.Tensor]:
    values = _dequantize(quant, scale)
    prefix = tuple(values.shape[:-1])
    blocks = values.shape[-1] // 64
    blocked = values.reshape(prefix + (blocks, 8, 2, 4))
    # Base scale 1.0 is a legal E6M2 value.  This is intentionally a minimal
    # fixture rather than a competitive quantizer.
    scale_factor = torch.ones(
        prefix + (blocks, 1, 1, 1), dtype=torch.float32, device=values.device
    )
    scale_lv2 = torch.ones(
        prefix + (blocks, 8, 1, 1), dtype=torch.float32, device=values.device
    )
    scale_lv3 = torch.ones(
        prefix + (blocks, 8, 2, 1), dtype=torch.float32, device=values.device
    )
    sign = torch.sign(blocked)
    mant = (blocked.abs() * 4.0).round().clamp_(0, 7) * 0.25
    return {
        "scale_factor": scale_factor,
        "scale_lv2": scale_lv2,
        "scale_lv3": scale_lv3,
        "sign": sign,
        "mant": mant,
    }


def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    del calib_activation_list
    return {
        "weight_params": _quantize(weight_quant, weight_scale),
        "activation_state": None,
    }


def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_state: Any,
) -> dict[str, torch.Tensor]:
    del activation_state
    return _quantize(activation_quant, activation_scale)


def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    del calib_qkv_list, q_num_heads, kv_num_heads, head_dim
    return {"q_state": None, "k_state": None, "v_state": None}


def hif4_dynamic_quantize_q(
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    q_num_heads: int,
    head_dim: int,
    q_state: Any,
) -> dict[str, torch.Tensor]:
    del q_num_heads, head_dim, q_state
    return _quantize(q_quant, q_scale)


def hif4_dynamic_quantize_k(
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    k_state: Any,
) -> dict[str, torch.Tensor]:
    del kv_num_heads, head_dim, k_state
    return _quantize(k_quant, k_scale)


def hif4_dynamic_quantize_v(
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    v_state: Any,
) -> dict[str, torch.Tensor]:
    del kv_num_heads, head_dim, v_state
    return _quantize(v_quant, v_scale)
