"""Attention-aware v2 candidate for NVFP4 -> HiF4 conversion.

Calibration-only diagonal second moments of the Q/K dot-product counterpart
drive an inverse Smooth-QK transform, followed by an identical orthonormal
Hadamard transform.  This preserves floating QK exactly while balancing and
mixing dimensions before the fixed-base HiF4 DP.  A guarded weighted-DP/base
search implementation remains available for controlled ablation, but is off
in the accepted state because it failed the causal-MSE and latency gates.  No
token-by-token attention matrix is formed during calibration or quantization.
"""

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
    if quant_float.ndim == 0 or scale_float.ndim == 0:
        raise ValueError("NVFP4 carrier and scale must have at least one dimension")
    channels = int(quant_float.shape[-1])
    if channels % blk_size:
        raise ValueError("the last dimension must be divisible by 16")
    expected = quant_float.shape[:-1] + (channels // blk_size,)
    if scale_float.shape != expected:
        raise ValueError("NVFP4 scale shape does not match carrier shape")
    blocked = quant_float.reshape(*quant_float.shape[:-1], -1, blk_size)
    return (blocked * scale_float.unsqueeze(-1)).flatten(-2).to(torch.bfloat16)


def _restore_as_hif4_blocks(
    quant_float: torch.Tensor, scale_float: torch.Tensor
) -> torch.Tensor:
    channels = int(quant_float.shape[-1])
    if channels % HIF4_BLOCK_SIZE:
        raise ValueError("the last dimension must be divisible by 64")
    expected = quant_float.shape[:-1] + (channels // NVFP4_BLOCK_SIZE,)
    if scale_float.shape != expected:
        raise ValueError("NVFP4 scale shape does not match carrier shape")
    prefix = quant_float.shape[:-1]
    blocks = channels // HIF4_BLOCK_SIZE
    q16 = quant_float.reshape(*prefix, blocks, 4, NVFP4_BLOCK_SIZE)
    s16 = scale_float.reshape(*prefix, blocks, 4, 1)
    restored = (q16 * s16).to(torch.bfloat16).to(torch.float32)
    return restored.reshape(*prefix, blocks, 8, 2, 4)


def _nearest_e6m2(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32).clamp(min=_E6M2_MIN, max=_E6M2_MAX).contiguous()
    bits = x.view(torch.int32)
    retained_lsb = (bits >> 21) & 1
    rounded_bits = (bits + 0x000FFFFF + retained_lsb) & ~0x001FFFFF
    return rounded_bits.view(torch.float32).clamp_(max=_E6M2_MAX)


def _e6m2_neighbourhood(base: torch.Tensor) -> torch.Tensor:
    """Return predecessor/base/successor finite E6M2 values per block."""
    bits = base.contiguous().view(torch.int32)
    offsets = torch.tensor(
        (-0x00200000, 0, 0x00200000), dtype=torch.int32, device=base.device
    )
    candidates = bits.unsqueeze(-1) + offsets
    candidates.clamp_(min=0x27800000, max=0x47400000)
    return candidates.contiguous().view(torch.float32)


def _reshape_sensitivity(
    sensitivity: torch.Tensor | None,
    source_shape: torch.Size,
    heads: int | None,
    head_dim: int | None,
    device: torch.device,
) -> torch.Tensor | None:
    """Broadcast head/dimension statistics to the input's verified layout."""
    if sensitivity is None or heads is None or head_dim is None:
        return None
    stats = sensitivity.to(device=device, dtype=torch.float32)
    if tuple(stats.shape) != (heads, head_dim):
        return None
    channels = int(source_shape[-1])
    if channels == heads * head_dim:
        return stats.reshape(channels)
    if channels != head_dim:
        return None

    # Explicit-head layouts (..., H, ..., D) are supported when a matching
    # prefix axis exists.  Use the rightmost such axis, which covers B,T,H,D
    # and B,H,T,D without hard-coding either layout.
    head_axes = [index for index, size in enumerate(source_shape[:-1]) if size == heads]
    if not head_axes:
        return stats.mean(dim=0)
    axis = head_axes[-1]
    shape = [1] * len(source_shape)
    shape[axis] = heads
    shape[-1] = head_dim
    return stats.reshape(shape)


def _blocked_sensitivity(
    sensitivity: torch.Tensor | None,
    restored: torch.Tensor,
    source_shape: torch.Size,
    heads: int | None,
    head_dim: int | None,
) -> torch.Tensor:
    weight = _reshape_sensitivity(
        sensitivity, source_shape, heads, head_dim, restored.device
    )
    if weight is None:
        return torch.ones_like(restored)
    # Broadcasting first keeps the mapping correct for flattened and explicit
    # head layouts; the subsequent view follows the exact HiF4 block layout.
    logical_shape = tuple(source_shape[:-1]) + (int(source_shape[-1]),)
    return torch.broadcast_to(weight, logical_shape).reshape_as(restored)


def _local_losses(
    absolute: torch.Tensor,
    base: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    losses: list[torch.Tensor] = []
    for multiplier in (1.0, 2.0, 4.0):
        denominator = base[..., None, None, None] * multiplier
        code = torch.round(absolute * (4.0 / denominator)).clamp_(0.0, 7.0)
        error = code.mul(denominator * 0.25).sub_(absolute).square_()
        losses.append((error * weight).sum(dim=-1))
    return losses[0], losses[1], losses[2]


def _hierarchy(
    absolute: torch.Tensor,
    base: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    loss0, loss1, loss2 = _local_losses(absolute, base, weight)
    path1 = torch.minimum(loss0, loss1).sum(dim=-1)
    path2 = torch.minimum(loss1, loss2).sum(dim=-1)
    lv2_bit = path2 < path1
    lv3_bit = torch.where(
        lv2_bit.unsqueeze(-1), loss2 < loss1, loss1 < loss0
    )
    block_loss = torch.where(lv2_bit, path2, path1).sum(dim=-1)
    return lv2_bit, lv3_bit, block_loss


def _selected_block_loss(
    absolute: torch.Tensor,
    base: torch.Tensor,
    lv2_bit: torch.Tensor,
    lv3_bit: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    level = (lv2_bit.to(torch.float32) + 1.0).unsqueeze(-1)
    level = level * (lv3_bit.to(torch.float32) + 1.0)
    denominator = base[..., None, None, None] * level.unsqueeze(-1)
    code = torch.round(absolute * (4.0 / denominator)).clamp_(0.0, 7.0)
    error = code.mul(denominator * 0.25).sub_(absolute).square_()
    return (error * weight).sum(dim=(-1, -2, -3))


def _quantize_hif4(
    quant_float: torch.Tensor,
    scale_float: torch.Tensor,
    sensitivity: torch.Tensor | None = None,
    heads: int | None = None,
    head_dim: int | None = None,
    search_radius: int = 0,
) -> HiF4Params:
    values = dequantize_nvfp4(quant_float, scale_float).to(torch.float32)
    return _quantize_values(
        values, sensitivity, heads, head_dim, search_radius
    )


def _quantize_values(
    values: torch.Tensor,
    sensitivity: torch.Tensor | None = None,
    heads: int | None = None,
    head_dim: int | None = None,
    search_radius: int = 0,
) -> HiF4Params:
    if values.shape[-1] % HIF4_BLOCK_SIZE:
        raise ValueError("the last dimension must be divisible by 64")
    source_shape = values.shape
    restored = values.to(torch.float32).reshape(
        *values.shape[:-1], values.shape[-1] // 64, 8, 2, 4
    )
    absolute = restored.abs()
    weight = _blocked_sensitivity(
        sensitivity, restored, source_shape, heads, head_dim
    )
    peak64 = absolute.amax(dim=(-1, -2, -3))
    raw_scale = (peak64.to(torch.bfloat16) * _BF16_ONE_OVER_SEVEN).to(torch.float32)
    base = _nearest_e6m2(raw_scale)

    if search_radius:
        plain_weight = torch.ones_like(restored)
        baseline_lv2, baseline_lv3, _ = _hierarchy(
            absolute, base, plain_weight
        )
        baseline_plain = _selected_block_loss(
            absolute, base, baseline_lv2, baseline_lv3, plain_weight
        )
        baseline_weighted = _selected_block_loss(
            absolute, base, baseline_lv2, baseline_lv3, weight
        )
        candidates = _e6m2_neighbourhood(base)
        candidate_losses: list[torch.Tensor] = []
        for index in range(3):
            _, _, loss = _hierarchy(absolute, candidates[..., index], weight)
            candidate_losses.append(loss)
        best = torch.stack(candidate_losses, dim=-1).argmin(dim=-1, keepdim=True)
        candidate_base = candidates.gather(-1, best).squeeze(-1)
        candidate_lv2, candidate_lv3, _ = _hierarchy(
            absolute, candidate_base, weight
        )
        candidate_plain = _selected_block_loss(
            absolute,
            candidate_base,
            candidate_lv2,
            candidate_lv3,
            plain_weight,
        )
        candidate_weighted = _selected_block_loss(
            absolute, candidate_base, candidate_lv2, candidate_lv3, weight
        )
        adopt = (candidate_weighted < baseline_weighted * 0.99) & (
            candidate_plain <= baseline_plain * 1.0025
        )
        base = torch.where(adopt, candidate_base, base)
        lv2_bit = torch.where(adopt.unsqueeze(-1), candidate_lv2, baseline_lv2)
        lv3_bit = torch.where(adopt.unsqueeze(-1).unsqueeze(-1), candidate_lv3, baseline_lv3)
    else:
        lv2_bit, lv3_bit, _ = _hierarchy(absolute, base, weight)
    scale_lv2 = lv2_bit.to(torch.float32).add_(1.0)
    scale_lv3 = lv3_bit.to(torch.float32).add_(1.0)
    total_scale = scale_lv2.unsqueeze(-1) * scale_lv3
    denominator = base[..., None, None, None] * total_scale.unsqueeze(-1)
    mant = torch.round(absolute * (4.0 / denominator)).clamp_(0.0, 7.0)
    mant.mul_(0.25)
    sign = torch.where(mant == 0, torch.zeros_like(restored), torch.sign(restored))
    return {
        "scale_factor": base.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
        "scale_lv2": scale_lv2.unsqueeze(-1).unsqueeze(-1),
        "scale_lv3": scale_lv3.unsqueeze(-1),
        "sign": sign,
        "mant": mant,
    }


def _head_second_moment(
    quant: torch.Tensor,
    scale: torch.Tensor,
    heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Extract E[x^2] per (head, head_dim) without assuming a token layout."""
    values = dequantize_nvfp4(quant, scale).to(torch.float32)
    if values.shape[-1] == heads * head_dim:
        headed = values.reshape(*values.shape[:-1], heads, head_dim)
        reduce_dims = tuple(range(headed.ndim - 2))
        return headed.square().mean(dim=reduce_dims)
    if values.shape[-1] != head_dim:
        raise ValueError("cannot identify Attention head/dimension layout")
    head_axes = [index for index, size in enumerate(values.shape[:-1]) if size == heads]
    if not head_axes:
        raise ValueError("explicit Attention layout has no identifiable head axis")
    head_axis = head_axes[-1]
    squared = values.square().movedim(head_axis, -2)
    reduce_dims = tuple(range(squared.ndim - 2))
    return squared.mean(dim=reduce_dims)


def _condition_sensitivity(moment: torch.Tensor) -> torch.Tensor:
    """Normalize per head and bound sampling noise/outlier leverage."""
    mean = moment.mean(dim=-1, keepdim=True).clamp_min(1.0e-20)
    return (moment / mean).clamp_(0.125, 8.0)


def _attention_hadamard(values: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
    """Apply a deterministic block-diagonal orthonormal Walsh transform."""
    if values.shape[-1] != heads * head_dim or head_dim % 64:
        raise ValueError("Attention tensor must be flattened [*, heads*head_dim]")
    original_shape = values.shape
    blocks = values.reshape(-1, heads, head_dim // 64, 64)
    transformed = blocks
    width = 1
    while width < 64:
        paired = transformed.reshape(*blocks.shape[:-1], -1, 2, width)
        left = paired[..., 0, :]
        right = paired[..., 1, :]
        transformed = torch.cat((left + right, left - right), dim=-1).reshape_as(blocks)
        width *= 2
    return (transformed * 0.125).reshape(original_shape)


def _transform_qk(
    values: torch.Tensor,
    smooth: torch.Tensor,
    heads: int,
    head_dim: int,
    divide: bool,
) -> torch.Tensor:
    headed = values.reshape(-1, heads, head_dim)
    local_smooth = smooth.to(device=values.device, dtype=torch.float32)
    headed = headed / local_smooth if divide else headed * local_smooth
    return _attention_hadamard(headed.reshape_as(values), heads, head_dim)


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
    if q_num_heads % kv_num_heads:
        raise ValueError("q_num_heads must be divisible by kv_num_heads")
    if head_dim not in (64, 128, 256):
        raise ValueError("head_dim must be 64, 128 or 256")
    if not calib_qkv_list:
        return {
            "q_state": {"version": 2, "search_radius": 0, "use_weighted": False},
            "k_state": {"version": 2, "search_radius": 0, "use_weighted": False},
            "v_state": {"version": 2, "search_radius": 0},
        }

    q_moments: list[torch.Tensor] = []
    k_moments: list[torch.Tensor] = []
    q_values_list: list[torch.Tensor] = []
    k_values_list: list[torch.Tensor] = []
    for item in calib_qkv_list:
        if isinstance(item, dict):
            if not all(name in item for name in ("q", "k", "v")):
                raise ValueError("Attention calibration dict must contain q/k/v")
            q_quant, q_scale = item["q"]
            k_quant, k_scale = item["k"]
        else:
            if len(item) < 6:
                raise ValueError("Attention calibration item must contain Q/K/V pairs")
            q_quant, q_scale, k_quant, k_scale = item[:4]
        q_values = dequantize_nvfp4(q_quant, q_scale).to(torch.float32)
        k_values = dequantize_nvfp4(k_quant, k_scale).to(torch.float32)
        q_values_list.append(q_values)
        k_values_list.append(k_values)
        q_moments.append(
            q_values.reshape(-1, q_num_heads, head_dim).square().mean(dim=0)
        )
        k_moments.append(
            k_values.reshape(-1, kv_num_heads, head_dim).square().mean(dim=0)
        )

    q_moment = torch.stack(q_moments).mean(dim=0)
    k_moment = torch.stack(k_moments).mean(dim=0)
    repeats = q_num_heads // kv_num_heads
    grouped_q = q_moment.reshape(kv_num_heads, repeats, head_dim).mean(dim=1)
    # Q/s and K*s preserve every unquantized dot product.  The fourth-root
    # moment ratio is the RMS analogue of the conservative alpha=0.25 rule.
    smooth_k = (grouped_q.clamp_min(1.0e-20) / k_moment.clamp_min(1.0e-20)).pow(0.125)
    smooth_k.clamp_(1.0 / 16.0, 16.0)
    smooth_q = smooth_k.repeat_interleave(repeats, dim=0)

    transformed_q_moments: list[torch.Tensor] = []
    transformed_k_moments: list[torch.Tensor] = []
    for q_values, k_values in zip(q_values_list, k_values_list):
        q_transformed = _transform_qk(
            q_values, smooth_q, q_num_heads, head_dim, True
        )
        k_transformed = _transform_qk(
            k_values, smooth_k, kv_num_heads, head_dim, False
        )
        transformed_q_moments.append(
            q_transformed.reshape(-1, q_num_heads, head_dim).square().mean(dim=0)
        )
        transformed_k_moments.append(
            k_transformed.reshape(-1, kv_num_heads, head_dim).square().mean(dim=0)
        )
    transformed_q = torch.stack(transformed_q_moments).mean(dim=0)
    transformed_k = torch.stack(transformed_k_moments).mean(dim=0)
    q_sensitivity = _condition_sensitivity(transformed_k).repeat_interleave(
        repeats, dim=0
    )
    grouped_transformed_q = transformed_q.reshape(
        kv_num_heads, repeats, head_dim
    ).mean(dim=1)
    k_sensitivity = _condition_sensitivity(grouped_transformed_q)
    return {
        "q_state": {
            "version": 2,
            "sensitivity": q_sensitivity,
            "smooth": smooth_q,
            "search_radius": 0,
            "use_weighted": False,
        },
        "k_state": {
            "version": 2,
            "sensitivity": k_sensitivity,
            "smooth": smooth_k,
            "search_radius": 0,
            "use_weighted": False,
        },
        "v_state": {"version": 2, "search_radius": 0},
    }


@torch.inference_mode()
def hif4_dynamic_quantize_q(
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    q_num_heads: int,
    head_dim: int,
    q_state: Any,
) -> HiF4Params:
    state = q_state if isinstance(q_state, dict) else {}
    values = dequantize_nvfp4(q_quant, q_scale).to(torch.float32)
    smooth = state.get("smooth")
    if isinstance(smooth, torch.Tensor):
        values = _transform_qk(values, smooth, q_num_heads, head_dim, True)
    sensitivity = state.get("sensitivity") if state.get("use_weighted") else None
    return _quantize_values(
        values,
        sensitivity,
        q_num_heads,
        head_dim,
        int(state.get("search_radius", 0)),
    )


@torch.inference_mode()
def hif4_dynamic_quantize_k(
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    k_state: Any,
) -> HiF4Params:
    state = k_state if isinstance(k_state, dict) else {}
    values = dequantize_nvfp4(k_quant, k_scale).to(torch.float32)
    smooth = state.get("smooth")
    if isinstance(smooth, torch.Tensor):
        values = _transform_qk(values, smooth, kv_num_heads, head_dim, False)
    sensitivity = state.get("sensitivity") if state.get("use_weighted") else None
    return _quantize_values(
        values,
        sensitivity,
        kv_num_heads,
        head_dim,
        int(state.get("search_radius", 0)),
    )


@torch.inference_mode()
def hif4_dynamic_quantize_v(
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    v_state: Any,
) -> HiF4Params:
    state = v_state if isinstance(v_state, dict) else {}
    return _quantize_hif4(
        v_quant,
        v_scale,
        None,
        kv_num_heads,
        head_dim,
        int(state.get("search_radius", 0)),
    )
