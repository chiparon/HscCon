"""V2 HiF4 candidate: local E6M2 base search plus exact hierarchy DP.

For every 64-value block this candidate evaluates the nominal ``amax / 7``
E6M2 base and its immediately adjacent representable E6M2 values.  The
level-2/level-3 hierarchy is solved exactly for every base; the block keeps the
base with minimum tensor reconstruction SSE.  The selected three-base path
fuses the candidate dimension to reduce kernel-launch overhead; a sequential
low-memory equivalent remains available as an ablation helper.
"""

from __future__ import annotations

from typing import Any, TypedDict

import torch


NVFP4_BLOCK_SIZE = 16
HIF4_BLOCK_SIZE = 64
_E6M2_MIN = 2.0**-48
_E6M2_MAX = 49152.0
_E6M2_BIT_STEP = 0x00200000
_BF16_ONE_OVER_SEVEN = 0.142578125
_MIN_RELATIVE_IMPROVEMENT = 0.01


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


def _restore_as_hif4_blocks(
    quant_float: torch.Tensor, scale_float: torch.Tensor
) -> torch.Tensor:
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
    blocks = channels // HIF4_BLOCK_SIZE
    q16 = quant_float.reshape(*prefix, blocks, 4, NVFP4_BLOCK_SIZE)
    s16 = scale_float.reshape(*prefix, blocks, 4, 1)
    restored = (q16 * s16).to(torch.bfloat16).to(torch.float32)
    return restored.reshape(*prefix, blocks, 8, 2, 4)


def _nearest_e6m2(x: torch.Tensor) -> torch.Tensor:
    """Round positive FP32 values to finite normal-only E6M2, ties-to-even."""
    x = x.to(torch.float32).clamp(min=_E6M2_MIN, max=_E6M2_MAX).contiguous()
    bits = x.view(torch.int32)
    retained_lsb = (bits >> 21) & 1
    rounded_bits = (bits + 0x000FFFFF + retained_lsb) & ~0x001FFFFF
    return rounded_bits.view(torch.float32).clamp_(max=_E6M2_MAX)


def _offset_e6m2(base: torch.Tensor, steps: int) -> torch.Tensor:
    """Move by an integer number of E6M2 codes, saturating at the bounds."""
    if not isinstance(steps, int) or steps == 0:
        raise ValueError("steps must be a nonzero integer")
    bits = base.contiguous().view(torch.int32)
    moved = bits + steps * _E6M2_BIT_STEP
    minimum_bits = torch.tensor(_E6M2_MIN, dtype=torch.float32).view(torch.int32)
    maximum_bits = torch.tensor(_E6M2_MAX, dtype=torch.float32).view(torch.int32)
    moved.clamp_(int(minimum_bits.item()), int(maximum_bits.item()))
    return moved.view(torch.float32)


def _loss_at_exponent(
    absolute: torch.Tensor, base_scale: torch.Tensor, exponent_scale: float
) -> torch.Tensor:
    denominator = base_scale[..., None, None, None] * exponent_scale
    code = torch.round(absolute * (4.0 / denominator)).clamp_(0.0, 7.0)
    code.mul_(denominator * 0.25).sub_(absolute).square_()
    return code.sum(dim=-1)


def _loss_at_exponent_many(
    absolute: torch.Tensor,
    candidate_scales: torch.Tensor,
    exponent_scale: float,
) -> torch.Tensor:
    """Evaluate one local exponent for all global bases in one tensor kernel."""
    denominator = candidate_scales[..., None, None, None] * exponent_scale
    code = torch.round(
        absolute.unsqueeze(-4) * (4.0 / denominator)
    ).clamp_(0.0, 7.0)
    code.mul_(denominator * 0.25).sub_(absolute.unsqueeze(-4)).square_()
    return code.sum(dim=-1)


def _solve_hierarchy(
    absolute: torch.Tensor, base_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-block SSE and exact lv2/lv3 decisions for one base."""
    loss0 = _loss_at_exponent(absolute, base_scale, 1.0)
    loss1 = _loss_at_exponent(absolute, base_scale, 2.0)
    loss2 = _loss_at_exponent(absolute, base_scale, 4.0)

    path_l2_1 = torch.minimum(loss0, loss1).sum(dim=-1)
    path_l2_2 = torch.minimum(loss1, loss2).sum(dim=-1)
    lv2_bit = path_l2_2 < path_l2_1
    lv3_if_l2_1 = loss1 < loss0
    lv3_if_l2_2 = loss2 < loss1
    lv3_bit = torch.where(lv2_bit.unsqueeze(-1), lv3_if_l2_2, lv3_if_l2_1)
    block_loss = torch.where(lv2_bit, path_l2_2, path_l2_1).sum(dim=-1)
    return block_loss, lv2_bit, lv3_bit


def _quantize_hif4_search(
    quant_float: torch.Tensor,
    scale_float: torch.Tensor,
    neighbor_steps: tuple[int, ...],
) -> HiF4Params:
    restored = _restore_as_hif4_blocks(quant_float, scale_float)
    absolute = restored.abs()
    peak64 = absolute.amax(dim=(-1, -2, -3))
    raw_scale = (peak64.to(torch.bfloat16) * _BF16_ONE_OVER_SEVEN).to(torch.float32)
    nominal = _nearest_e6m2(raw_scale)

    # Nominal first makes exact ties retain V1.  Candidate processing is
    # sequential: fixed complexity, low peak memory, and no data-dependent loop.
    best_scale = nominal
    baseline_loss, baseline_lv2, baseline_lv3 = _solve_hierarchy(absolute, nominal)
    best_loss = baseline_loss
    best_lv2 = baseline_lv2
    best_lv3 = baseline_lv3
    for step in neighbor_steps:
        candidate_scale = _offset_e6m2(nominal, step)
        candidate_loss, candidate_lv2, candidate_lv3 = _solve_hierarchy(
            absolute, candidate_scale
        )
        better = candidate_loss < best_loss
        best_loss = torch.where(better, candidate_loss, best_loss)
        best_scale = torch.where(better, candidate_scale, best_scale)
        best_lv2 = torch.where(better.unsqueeze(-1), candidate_lv2, best_lv2)
        best_lv3 = torch.where(
            better.unsqueeze(-1).unsqueeze(-1), candidate_lv3, best_lv3
        )

    # A local tensor-SSE win can be too small to survive multiplication or
    # softmax.  Require a material per-block win; otherwise preserve every V1
    # field, including its tie-breaking, bit for bit.
    accept = best_loss < baseline_loss * (1.0 - _MIN_RELATIVE_IMPROVEMENT)
    best_scale = torch.where(accept, best_scale, nominal)
    best_lv2 = torch.where(accept.unsqueeze(-1), best_lv2, baseline_lv2)
    best_lv3 = torch.where(
        accept.unsqueeze(-1).unsqueeze(-1), best_lv3, baseline_lv3
    )

    scale_lv2 = best_lv2.to(torch.float32).add_(1.0)
    scale_lv3 = best_lv3.to(torch.float32).add_(1.0)
    total_scale = scale_lv2.unsqueeze(-1) * scale_lv3
    denominator = best_scale[..., None, None, None] * total_scale.unsqueeze(-1)
    mant = torch.round(absolute * (4.0 / denominator)).clamp_(0.0, 7.0)
    mant.mul_(0.25)
    sign = torch.where(mant == 0, torch.zeros_like(restored), torch.sign(restored))

    return {
        "scale_factor": best_scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
        "scale_lv2": scale_lv2.unsqueeze(-1).unsqueeze(-1),
        "scale_lv3": scale_lv3.unsqueeze(-1),
        "sign": sign,
        "mant": mant,
    }


def _quantize_hif4(
    quant_float: torch.Tensor, scale_float: torch.Tensor
) -> HiF4Params:
    """Guarded three-base search used by the selected operand policy."""
    return _quantize_hif4_search(quant_float, scale_float, (-1, 1))


def _quantize_hif4_five(
    quant_float: torch.Tensor, scale_float: torch.Tensor
) -> HiF4Params:
    """Five-base ablation retained for reproducible neighborhood evaluation."""
    return _quantize_hif4_search(quant_float, scale_float, (-1, 1, -2, 2))


def _quantize_hif4_nominal(
    quant_float: torch.Tensor, scale_float: torch.Tensor
) -> HiF4Params:
    """V1-compatible single-base path used for sensitive Q/K operands."""
    return _quantize_hif4_search(quant_float, scale_float, ())


def _quantize_hif4_fused_three(
    quant_float: torch.Tensor, scale_float: torch.Tensor
) -> HiF4Params:
    """Three-base ablation that trades scratch memory for fewer kernel launches."""
    restored = _restore_as_hif4_blocks(quant_float, scale_float)
    absolute = restored.abs()
    peak64 = absolute.amax(dim=(-1, -2, -3))
    raw_scale = (peak64.to(torch.bfloat16) * _BF16_ONE_OVER_SEVEN).to(torch.float32)
    nominal = _nearest_e6m2(raw_scale)
    candidates = torch.stack(
        (nominal, _offset_e6m2(nominal, -1), _offset_e6m2(nominal, 1)),
        dim=-1,
    )

    loss0 = _loss_at_exponent_many(absolute, candidates, 1.0)
    loss1 = _loss_at_exponent_many(absolute, candidates, 2.0)
    loss2 = _loss_at_exponent_many(absolute, candidates, 4.0)
    path_l2_1 = torch.minimum(loss0, loss1).sum(dim=-1)
    path_l2_2 = torch.minimum(loss1, loss2).sum(dim=-1)
    lv2_bits = path_l2_2 < path_l2_1
    lv3_bits = torch.where(
        lv2_bits.unsqueeze(-1), loss2 < loss1, loss1 < loss0
    )
    block_losses = torch.where(lv2_bits, path_l2_2, path_l2_1).sum(dim=-1)

    best_loss, selected = block_losses.min(dim=-1)
    baseline_loss = block_losses[..., 0]
    accept = best_loss < baseline_loss * (1.0 - _MIN_RELATIVE_IMPROVEMENT)
    selected = torch.where(accept, selected, torch.zeros_like(selected))
    best_scale = torch.gather(candidates, -1, selected.unsqueeze(-1)).squeeze(-1)

    lv2_index = selected.unsqueeze(-1).unsqueeze(-1).expand(
        *selected.shape, 1, 8
    )
    best_lv2 = torch.gather(lv2_bits, -2, lv2_index).squeeze(-2)
    lv3_index = selected.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
        *selected.shape, 1, 8, 2
    )
    best_lv3 = torch.gather(lv3_bits, -3, lv3_index).squeeze(-3)

    scale_lv2 = best_lv2.to(torch.float32).add_(1.0)
    scale_lv3 = best_lv3.to(torch.float32).add_(1.0)
    total_scale = scale_lv2.unsqueeze(-1) * scale_lv3
    denominator = best_scale[..., None, None, None] * total_scale.unsqueeze(-1)
    mant = torch.round(absolute * (4.0 / denominator)).clamp_(0.0, 7.0)
    mant.mul_(0.25)
    sign = torch.where(mant == 0, torch.zeros_like(restored), torch.sign(restored))
    return {
        "scale_factor": best_scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
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
        "weight_params": _quantize_hif4_fused_three(weight_quant, weight_scale),
        "activation_state": {"version": 3, "base_search": 3},
    }


@torch.inference_mode()
def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_state: Any,
) -> HiF4Params:
    del activation_state
    return _quantize_hif4_fused_three(activation_quant, activation_scale)


@torch.inference_mode()
def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    del calib_qkv_list, q_num_heads, kv_num_heads, head_dim
    return {
        "q_state": {"version": 3, "base_search": 1},
        "k_state": {"version": 3, "base_search": 1},
        "v_state": {"version": 3, "base_search": 3},
    }


@torch.inference_mode()
def hif4_dynamic_quantize_q(
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    q_num_heads: int,
    head_dim: int,
    q_state: Any,
) -> HiF4Params:
    del q_num_heads, head_dim, q_state
    return _quantize_hif4_nominal(q_quant, q_scale)


@torch.inference_mode()
def hif4_dynamic_quantize_k(
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    k_state: Any,
) -> HiF4Params:
    del kv_num_heads, head_dim, k_state
    return _quantize_hif4_nominal(k_quant, k_scale)


@torch.inference_mode()
def hif4_dynamic_quantize_v(
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    v_state: Any,
) -> HiF4Params:
    del kv_num_heads, head_dim, v_state
    return _quantize_hif4_fused_three(v_quant, v_scale)
