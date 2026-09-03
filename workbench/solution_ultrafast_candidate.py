"""Allocation-reduced, fixed-work NVFP4 -> HiF4 heuristic candidate.

This performance variant intentionally assumes valid competition inputs on the
hot path.  It restores once, emits the sign, and then reuses the restored FP32
buffer in-place for absolute values and finally for S1P2 mantissas.
"""

from __future__ import annotations

from typing import Any, TypedDict

import torch


NVFP4_BLOCK_SIZE = 16
HIF4_BLOCK_SIZE = 64
_E6M2_MIN = 2.0**-48
_E6M2_MAX = 49152.0


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
    """Compatibility helper; the six competition functions use the fast path."""
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


def _nearest_e6m2(x: torch.Tensor) -> torch.Tensor:
    """Finite unsigned E6M2 rounding via FP32 significand bit operations."""
    x.clamp_(min=_E6M2_MIN, max=_E6M2_MAX)
    bits = x.contiguous().view(torch.int32)
    retained_lsb = (bits >> 21) & 1
    bits.add_(0x000FFFFF).add_(retained_lsb).bitwise_and_(~0x001FFFFF)
    return bits.view(torch.float32).clamp_(max=_E6M2_MAX)


def _quantize_hif4(
    quant_float: torch.Tensor, scale_float: torch.Tensor
) -> HiF4Params:
    prefix = quant_float.shape[:-1]
    blocks = quant_float.shape[-1] // HIF4_BLOCK_SIZE

    q16 = quant_float.reshape(*prefix, blocks, 4, NVFP4_BLOCK_SIZE)
    s16 = scale_float.reshape(*prefix, blocks, 4, 1)
    work = (q16 * s16).to(torch.bfloat16).to(torch.float32)
    work = work.reshape(*prefix, blocks, 8, 2, 4)

    sign = torch.sign(work)
    work.abs_()
    peak4 = work.amax(dim=-1)
    peak8 = peak4.amax(dim=-1)
    peak64 = peak8.amax(dim=-1)

    # peak64 is an owned reduction result and becomes the E6M2 result in-place.
    peak64.mul_(1.0 / 7.0)
    scale_factor = _nearest_e6m2(peak64)

    lv2_bit = peak8 >= scale_factor.unsqueeze(-1) * 4.0
    scale_lv2 = lv2_bit.to(torch.float32).add_(1.0)
    lv3_bit = peak4 >= (
        scale_factor.unsqueeze(-1).unsqueeze(-1)
        * scale_lv2.unsqueeze(-1)
        * 2.0
    )
    scale_lv3 = lv3_bit.to(torch.float32).add_(1.0)

    denominator = (
        scale_factor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        * scale_lv2.unsqueeze(-1).unsqueeze(-1)
        * scale_lv3.unsqueeze(-1)
    )
    # Reuse the restored buffer as the final mantissa tensor.
    work.div_(denominator).mul_(4.0).add_(0.5).floor_().clamp_(0.0, 7.0)
    work.mul_(0.25)

    return {
        "scale_factor": scale_factor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
        "scale_lv2": scale_lv2.unsqueeze(-1).unsqueeze(-1),
        "scale_lv3": scale_lv3.unsqueeze(-1),
        "sign": sign,
        "mant": work,
    }


@torch.inference_mode()
def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    return {
        "weight_params": _quantize_hif4(weight_quant, weight_scale),
        "activation_state": None,
    }


@torch.inference_mode()
def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_state: Any,
) -> HiF4Params:
    return _quantize_hif4(activation_quant, activation_scale)


@torch.inference_mode()
def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    return {"q_state": None, "k_state": None, "v_state": None}


@torch.inference_mode()
def hif4_dynamic_quantize_q(
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    q_num_heads: int,
    head_dim: int,
    q_state: Any,
) -> HiF4Params:
    return _quantize_hif4(q_quant, q_scale)


@torch.inference_mode()
def hif4_dynamic_quantize_k(
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    k_state: Any,
) -> HiF4Params:
    return _quantize_hif4(k_quant, k_scale)


@torch.inference_mode()
def hif4_dynamic_quantize_v(
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    v_state: Any,
) -> HiF4Params:
    return _quantize_hif4(v_quant, v_scale)
