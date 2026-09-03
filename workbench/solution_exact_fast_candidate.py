"""Exact dynamic-programming HiF4 candidate with a fixed, vectorized fast path.

The slow reference enumerates all eight legal (level-2, level-3a, level-3b)
layouts for every eight-value group.  This implementation evaluates the three
possible *total* micro-exponents once per four-value group, then solves the same
choice by a two-path dynamic program.  There are no loops over data blocks and
no candidate dimension containing copies of the full input tensor.
"""

from __future__ import annotations

from typing import Any, TypedDict

import torch


NVFP4_BLOCK_SIZE = 16
HIF4_BLOCK_SIZE = 64
_E6M2_MIN = 2.0**-48
_E6M2_MAX = 49152.0
# BF16 representation of 1/7 used by the exhaustive reference.
_BF16_ONE_OVER_SEVEN = 0.142578125


class HiF4Params(TypedDict):
    scale_factor: torch.Tensor
    scale_lv2: torch.Tensor
    scale_lv3: torch.Tensor
    sign: torch.Tensor
    mant: torch.Tensor


def dequantize_nvfp4(
    quant_float: torch.Tensor,
    scale_float: torch.Tensor,
    blk_size: int = NVFP4_BLOCK_SIZE,
) -> torch.Tensor:
    """Restore an NVFP4 carrier and its per-block scales to BF16."""
    if quant_float.ndim == 0 or scale_float.ndim == 0:
        raise ValueError("NVFP4 carrier and scale must have at least one dimension")
    if blk_size <= 0:
        raise ValueError(f"block size must be positive, got {blk_size}")
    channels = int(quant_float.shape[-1])
    if channels % blk_size:
        raise ValueError(
            f"last dimension {channels} is not divisible by block size {blk_size}"
        )
    expected = quant_float.shape[:-1] + (channels // blk_size,)
    if scale_float.shape != expected:
        raise ValueError(
            f"scale shape {tuple(scale_float.shape)} does not match expected "
            f"shape {tuple(expected)}"
        )
    x = quant_float.reshape(*quant_float.shape[:-1], -1, blk_size)
    return (x * scale_float.unsqueeze(-1)).flatten(-2).to(torch.bfloat16)


def _restore_as_hif4_blocks(
    quant_float: torch.Tensor, scale_float: torch.Tensor
) -> torch.Tensor:
    """Restore directly into ``(..., C/64, 8, 2, 4)``."""
    if quant_float.ndim == 0 or scale_float.ndim == 0:
        raise ValueError("NVFP4 carrier and scale must have at least one dimension")
    channels = int(quant_float.shape[-1])
    if channels % HIF4_BLOCK_SIZE:
        raise ValueError(
            f"last dimension {channels} is not divisible by {HIF4_BLOCK_SIZE}"
        )
    expected = quant_float.shape[:-1] + (channels // NVFP4_BLOCK_SIZE,)
    if scale_float.shape != expected:
        raise ValueError(
            f"scale shape {tuple(scale_float.shape)} does not match expected "
            f"shape {tuple(expected)}"
        )

    prefix = quant_float.shape[:-1]
    n64 = channels // HIF4_BLOCK_SIZE
    q16 = quant_float.reshape(*prefix, n64, 4, NVFP4_BLOCK_SIZE)
    s16 = scale_float.reshape(*prefix, n64, 4, 1)
    restored = (q16 * s16).to(torch.bfloat16).to(torch.float32)
    return restored.reshape(*prefix, n64, 8, 2, 4)


def _nearest_e6m2(x: torch.Tensor) -> torch.Tensor:
    """Round positive FP32 values to finite normal-only E6M2, ties-to-even."""
    x = x.to(torch.float32).clamp(min=_E6M2_MIN, max=_E6M2_MAX).contiguous()
    bits = x.view(torch.int32)
    retained_lsb = (bits >> 21) & 1
    rounded_bits = (bits + 0x000FFFFF + retained_lsb) & ~0x001FFFFF
    return rounded_bits.view(torch.float32).clamp_(max=_E6M2_MAX)


def _loss_at_exponent(
    absolute: torch.Tensor, base_scale: torch.Tensor, exponent_scale: float
) -> torch.Tensor:
    """Per-four-value reconstruction SSE for one total exponent."""
    denominator = base_scale[..., None, None, None] * exponent_scale
    code = torch.round(absolute * (4.0 / denominator)).clamp_(0.0, 7.0)
    code.mul_(denominator * 0.25).sub_(absolute).square_()
    return code.sum(dim=-1)


def _quantize_hif4(
    quant_float: torch.Tensor, scale_float: torch.Tensor
) -> HiF4Params:
    """Select the exact minimum-SSE hierarchy with an algebraically reduced DP."""
    x = _restore_as_hif4_blocks(quant_float, scale_float)
    absolute = x.abs()
    peak64 = absolute.amax(dim=(-1, -2, -3))
    raw_scale = (peak64.to(torch.bfloat16) * _BF16_ONE_OVER_SEVEN).to(
        torch.float32
    )
    scale_factor = _nearest_e6m2(raw_scale)

    # Only three total exponent values exist: lv2_bit + lv3_bit in {0,1,2}.
    # Each call holds one payload-sized temporary, rather than materializing a
    # 3x or 8x candidate tensor.
    loss0 = _loss_at_exponent(absolute, scale_factor, 1.0)
    loss1 = _loss_at_exponent(absolute, scale_factor, 2.0)
    loss2 = _loss_at_exponent(absolute, scale_factor, 4.0)

    # For lv2=1, each of the two lv3 branches independently chooses e=0/1.
    # For lv2=2 it chooses e=1/2.  Strict comparisons ensure every exact tie
    # selects the smaller exponent, matching the reference enumeration order.
    path_l2_1 = torch.minimum(loss0, loss1).sum(dim=-1)
    path_l2_2 = torch.minimum(loss1, loss2).sum(dim=-1)
    lv2_bit = path_l2_2 < path_l2_1
    scale_lv2 = lv2_bit.to(torch.float32).add_(1.0)

    lv3_if_l2_1 = loss1 < loss0
    lv3_if_l2_2 = loss2 < loss1
    lv3_bit = torch.where(lv2_bit.unsqueeze(-1), lv3_if_l2_2, lv3_if_l2_1)
    scale_lv3 = lv3_bit.to(torch.float32).add_(1.0)

    total_scale = scale_lv2.unsqueeze(-1) * scale_lv3
    denominator = scale_factor[..., None, None, None] * total_scale.unsqueeze(-1)
    mant = torch.round(absolute * (4.0 / denominator)).clamp_(0.0, 7.0)
    mant.mul_(0.25)
    sign = torch.where(mant == 0, torch.zeros_like(x), torch.sign(x))

    return {
        "scale_factor": scale_factor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
        "scale_lv2": scale_lv2.unsqueeze(-1).unsqueeze(-1),
        "scale_lv3": scale_lv3.unsqueeze(-1),
        "sign": sign,
        "mant": mant,
    }


@torch.inference_mode()
def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    del calib_activation_list
    return {
        "weight_params": _quantize_hif4(weight_quant, weight_scale),
        "activation_state": {"version": 2},
    }


@torch.inference_mode()
def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_state: Any,
) -> HiF4Params:
    del activation_state
    return _quantize_hif4(activation_quant, activation_scale)


@torch.inference_mode()
def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    del calib_qkv_list, q_num_heads, kv_num_heads, head_dim
    return {"q_state": {"version": 2}, "k_state": {"version": 2}, "v_state": {"version": 2}}


@torch.inference_mode()
def hif4_dynamic_quantize_q(
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    q_num_heads: int,
    head_dim: int,
    q_state: Any,
) -> HiF4Params:
    del q_num_heads, head_dim, q_state
    return _quantize_hif4(q_quant, q_scale)


@torch.inference_mode()
def hif4_dynamic_quantize_k(
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    k_state: Any,
) -> HiF4Params:
    del kv_num_heads, head_dim, k_state
    return _quantize_hif4(k_quant, k_scale)


@torch.inference_mode()
def hif4_dynamic_quantize_v(
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    v_state: Any,
) -> HiF4Params:
    del kv_num_heads, head_dim, v_state
    return _quantize_hif4(v_quant, v_scale)
