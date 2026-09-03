"""Fast, fixed-work NVFP4 -> HiF4 submission candidate.

This file is intentionally self-contained so it can be copied to ``solution.py``
for submission.  The conversion has no Python loop over tensor blocks and no
data-dependent candidate search.  Calibration states contain only Python scalar
data, making them portable and keeping the online path free of CPU/device copies.
"""

from __future__ import annotations

from typing import Any, TypedDict

import torch


NVFP4_BLOCK_SIZE = 16
HIF4_BLOCK_SIZE = 64
_E6M2_MIN = 2.0**-48
_E6M2_MAX = (1.0 + 2.0 / 4.0) * (2.0**15)


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
    """Restore directly into ``(..., C/64, 8, 2, 4)`` without a flat copy."""
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

    # Four source blocks make one HiF4 block.  The product is rounded to BF16,
    # exactly as dequantize_nvfp4 does, and widened once for the reductions.
    prefix = quant_float.shape[:-1]
    n64 = channels // HIF4_BLOCK_SIZE
    q16 = quant_float.reshape(*prefix, n64, 4, NVFP4_BLOCK_SIZE)
    s16 = scale_float.reshape(*prefix, n64, 4, 1)
    restored = (q16 * s16).to(torch.bfloat16).to(torch.float32)
    return restored.reshape(*prefix, n64, 8, 2, 4)


def _nearest_e6m2(x: torch.Tensor) -> torch.Tensor:
    """Return exact finite values from unsigned normal-only E6M2.

    E6M2 uses exponent bias 48, unbiased exponents [-48, 15], a hidden leading
    one, and two fraction bits.  Code 0xff is NaN, so the largest finite value
    is ``1.5 * 2**15``.  Clearing the low 21 FP32 significand bits implements
    round-to-nearest-even with a few integer operations (and no log/frexp/pow).
    """
    x = x.to(torch.float32).clamp(min=_E6M2_MIN, max=_E6M2_MAX).contiguous()
    bits = x.view(torch.int32)
    retained_lsb = (bits >> 21) & 1
    rounded_bits = (bits + 0x000FFFFF + retained_lsb) & ~0x001FFFFF
    return rounded_bits.view(torch.float32).clamp_(max=_E6M2_MAX)


def _quantize_hif4(
    quant_float: torch.Tensor,
    scale_float: torch.Tensor,
    scale_multiplier: float = 1.0,
) -> HiF4Params:
    """Vectorized Algorithm-1-style hierarchy selection and S1P2 rounding."""
    x = _restore_as_hif4_blocks(quant_float, scale_float)
    abs_x = x.abs()

    # The input is already laid out as (..., block64, group8, group4, value).
    # Keeping both reduction axes gives the two micro-scale levels directly.
    peak4 = abs_x.amax(dim=-1)
    peak8 = peak4.amax(dim=-1)
    peak64 = peak8.amax(dim=-1)

    # The E1/E1 hierarchy represents a maximum payload magnitude of 7.
    # A scalar multiplier is available for inexpensive calibration policies.
    scale_factor = _nearest_e6m2(peak64 * (float(scale_multiplier) / 7.0))

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
    # Magnitudes are non-negative, so floor(y + .5) is round-half-away.
    mant = torch.floor(abs_x * (4.0 / denominator) + 0.5).clamp_(0.0, 7.0)
    mant.mul_(0.25)
    sign = torch.sign(x)

    return {
        "scale_factor": scale_factor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
        "scale_lv2": scale_lv2.unsqueeze(-1).unsqueeze(-1),
        "scale_lv3": scale_lv3.unsqueeze(-1),
        "sign": sign,
        "mant": mant,
    }


def _state_multiplier(state: Any) -> float:
    """Read the only online tuning scalar while accepting absent/older state."""
    if isinstance(state, dict):
        value = state.get("scale_multiplier", 1.0)
        if isinstance(value, (int, float)):
            return float(value)
    return 1.0


@torch.inference_mode()
def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    """Quantize the static weight and emit a zero-copy scalar online state."""
    del calib_activation_list
    return {
        "weight_params": _quantize_hif4(weight_quant, weight_scale),
        "activation_state": {"version": 1, "scale_multiplier": 1.0},
    }


@torch.inference_mode()
def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_state: Any,
) -> HiF4Params:
    return _quantize_hif4(
        activation_quant,
        activation_scale,
        _state_multiplier(activation_state),
    )


@torch.inference_mode()
def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    """Emit portable scalar-only Q/K/V states; calibration has constant work."""
    del calib_qkv_list, q_num_heads, kv_num_heads, head_dim
    state = {"version": 1, "scale_multiplier": 1.0}
    # Do not alias the dictionaries: some harnesses serialize or mutate them.
    return {"q_state": dict(state), "k_state": dict(state), "v_state": dict(state)}


@torch.inference_mode()
def hif4_dynamic_quantize_q(
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    q_num_heads: int,
    head_dim: int,
    q_state: Any,
) -> HiF4Params:
    del q_num_heads, head_dim
    return _quantize_hif4(q_quant, q_scale, _state_multiplier(q_state))


@torch.inference_mode()
def hif4_dynamic_quantize_k(
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    k_state: Any,
) -> HiF4Params:
    del kv_num_heads, head_dim
    return _quantize_hif4(k_quant, k_scale, _state_multiplier(k_state))


@torch.inference_mode()
def hif4_dynamic_quantize_v(
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    v_state: Any,
) -> HiF4Params:
    del kv_num_heads, head_dim
    return _quantize_hif4(v_quant, v_scale, _state_multiplier(v_state))
