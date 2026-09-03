"""Allocation-reduced exact-DP NVFP4 -> HiF4 candidate."""

from __future__ import annotations

from typing import Any, TypedDict

import torch


NVFP4_BLOCK_SIZE = 16
HIF4_BLOCK_SIZE = 64
_E6M2_MIN = 2.0**-48
_E6M2_MAX = 49152.0
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
    """Compatibility helper; competition calls use the fused block path."""
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
    blocked = quant_float.reshape(*quant_float.shape[:-1], -1, blk_size)
    return (blocked * scale_float.unsqueeze(-1)).flatten(-2).to(torch.bfloat16)


def _nearest_e6m2_in_place(x: torch.Tensor) -> torch.Tensor:
    """Round an owned FP32 reduction buffer to finite E6M2 in place."""
    x.clamp_(min=_E6M2_MIN, max=_E6M2_MAX)
    bits = x.contiguous().view(torch.int32)
    retained_lsb = (bits >> 21) & 1
    bits.add_(0x000FFFFF).add_(retained_lsb).bitwise_and_(~0x001FFFFF)
    return bits.view(torch.float32).clamp_(max=_E6M2_MAX)


def _loss_at_scale(
    absolute: torch.Tensor, base_scale: torch.Tensor, exponent_scale: float
) -> torch.Tensor:
    """Return FP32 reconstruction SSE for each four-value subgroup."""
    denominator = base_scale[..., None, None, None] * exponent_scale
    error = absolute * (4.0 / denominator)
    error.round_().clamp_(0.0, 7.0)
    error.mul_(denominator * 0.25).sub_(absolute).square_()
    return error.sum(dim=-1)


def _quantize_hif4(
    quant_float: torch.Tensor, scale_float: torch.Tensor
) -> HiF4Params:
    # The official harness supplies valid contiguous shapes.  Four NVFP4
    # blocks are restored directly into one 64-value HiF4 hierarchy.
    prefix = quant_float.shape[:-1]
    blocks = quant_float.shape[-1] // HIF4_BLOCK_SIZE
    q16 = quant_float.reshape(*prefix, blocks, 4, NVFP4_BLOCK_SIZE)
    s16 = scale_float.reshape(*prefix, blocks, 4, 1)
    work = (q16 * s16).to(torch.bfloat16).to(torch.float32)
    work = work.reshape(*prefix, blocks, 8, 2, 4)

    sign = torch.sign(work)
    work.abs_()
    peak64 = work.amax(dim=(-1, -2, -3))
    raw_scale = peak64.to(torch.bfloat16)
    raw_scale.mul_(_BF16_ONE_OVER_SEVEN)
    scale_factor = _nearest_e6m2_in_place(raw_scale.to(torch.float32))

    # Three effective exponents replace eight full hierarchy candidates.
    loss0 = _loss_at_scale(work, scale_factor, 1.0)
    loss1 = _loss_at_scale(work, scale_factor, 2.0)
    loss2 = _loss_at_scale(work, scale_factor, 4.0)

    lv3_if_l2_1 = loss1 < loss0
    lv3_if_l2_2 = loss2 < loss1
    torch.minimum(loss0, loss1, out=loss0)
    torch.minimum(loss1, loss2, out=loss1)
    path_l2_1 = loss0.sum(dim=-1)
    path_l2_2 = loss1.sum(dim=-1)
    lv2_bit = path_l2_2 < path_l2_1

    scale_lv2 = lv2_bit.to(torch.float32).add_(1.0)
    lv3_bit = torch.where(lv2_bit.unsqueeze(-1), lv3_if_l2_2, lv3_if_l2_1)
    scale_lv3 = lv3_bit.to(torch.float32).add_(1.0)

    denominator = (
        scale_factor[..., None, None, None]
        * scale_lv2.unsqueeze(-1).unsqueeze(-1)
        * scale_lv3.unsqueeze(-1)
    )
    # Reuse the restored payload buffer as the final mantissa output.
    work.div_(denominator).mul_(4.0).round_().clamp_(0.0, 7.0).mul_(0.25)
    sign.masked_fill_(work == 0, 0.0)

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
