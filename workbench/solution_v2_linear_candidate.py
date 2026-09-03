"""Linear-aware HiF4 v2 candidate.

The calibration path uses only operand-local statistics:

* activation second moments weight the static weight conversion;
* column energy of the already converted weight weights online activations.

No activation/weight matrix product is formed by the quantizer.  The weighted
losses are diagonal approximations to Linear output MSE and stay separable over
the existing HiF4 64/8/4 hierarchy.
"""

from __future__ import annotations

from typing import Any, Mapping, TypedDict

import torch


NVFP4_BLOCK_SIZE = 16
HIF4_BLOCK_SIZE = 64
_E6M2_MIN = 2.0**-48
_E6M2_MAX = 49152.0
_E6M2_MIN_BITS = 0x27800000
_E6M2_MAX_BITS = 0x47400000
_E6M2_STEP_BITS = 0x00200000
_BF16_ONE_OVER_SEVEN = 0.142578125
_WEIGHT_BASE_OFFSETS = (0, 1)
_ACTIVATION_BASE_OFFSETS = (0,)
_USE_WEIGHT_IMPORTANCE = True
_USE_ACTIVATION_IMPORTANCE = True
_IMPORTANCE_MIN = 0.01
_IMPORTANCE_MAX = 64.0
_WEIGHT_IMPORTANCE_POWER = 0.5
_ACTIVATION_IMPORTANCE_POWER = 0.5


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
    x = x.to(torch.float32).clamp(min=_E6M2_MIN, max=_E6M2_MAX).contiguous()
    bits = x.view(torch.int32)
    retained_lsb = (bits >> 21) & 1
    rounded_bits = (bits + 0x000FFFFF + retained_lsb) & ~0x001FFFFF
    return rounded_bits.view(torch.float32).clamp_(max=_E6M2_MAX)


def _e6m2_neighbor(base: torch.Tensor, offset: int) -> torch.Tensor:
    """Return an adjacent legal finite E6M2 scale without host synchronization."""
    bits = base.contiguous().view(torch.int32)
    shifted = (bits + offset * _E6M2_STEP_BITS).clamp(
        _E6M2_MIN_BITS, _E6M2_MAX_BITS
    )
    return shifted.view(torch.float32)


def _prepare_importance(
    importance: torch.Tensor | None, restored: torch.Tensor
) -> torch.Tensor | None:
    if importance is None:
        return None
    expected = restored.shape[-4:]
    if importance.numel() != expected.numel():
        raise ValueError(
            f"importance has {importance.numel()} elements, expected {expected.numel()}"
        )
    return importance.to(device=restored.device, dtype=torch.float32).reshape(expected)


def _loss_at_exponent(
    absolute: torch.Tensor,
    base_scale: torch.Tensor,
    exponent_scale: float,
    importance: torch.Tensor | None,
) -> torch.Tensor:
    denominator = base_scale[..., None, None, None] * exponent_scale
    reconstructed = torch.round(absolute * (4.0 / denominator)).clamp_(0.0, 7.0)
    reconstructed.mul_(denominator * 0.25).sub_(absolute).square_()
    if importance is not None:
        reconstructed.mul_(importance)
    return reconstructed.sum(dim=-1)


def _solve_hierarchy(
    absolute: torch.Tensor,
    base_scale: torch.Tensor,
    importance: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    loss0 = _loss_at_exponent(absolute, base_scale, 1.0, importance)
    loss1 = _loss_at_exponent(absolute, base_scale, 2.0, importance)
    loss2 = _loss_at_exponent(absolute, base_scale, 4.0, importance)
    path_l2_1 = torch.minimum(loss0, loss1).sum(dim=-1)
    path_l2_2 = torch.minimum(loss1, loss2).sum(dim=-1)
    lv2_bit = path_l2_2 < path_l2_1
    lv3_if_l2_1 = loss1 < loss0
    lv3_if_l2_2 = loss2 < loss1
    lv3_bit = torch.where(lv2_bit.unsqueeze(-1), lv3_if_l2_2, lv3_if_l2_1)
    # ``base_scale`` is shared by the whole 64-value block, so base search
    # compares the sum over all eight level-2 nodes.
    best_loss = torch.where(lv2_bit, path_l2_2, path_l2_1).sum(dim=-1)
    return best_loss, lv2_bit, lv3_bit


def _quantize_hif4(
    quant_float: torch.Tensor,
    scale_float: torch.Tensor,
    importance: torch.Tensor | None = None,
    base_offsets: tuple[int, ...] = (0,),
) -> HiF4Params:
    restored = _restore_as_hif4_blocks(quant_float, scale_float)
    absolute = restored.abs()
    importance = _prepare_importance(importance, restored)
    peak64 = absolute.amax(dim=(-1, -2, -3))
    raw_scale = (peak64.to(torch.bfloat16) * _BF16_ONE_OVER_SEVEN).to(torch.float32)
    nominal_base = _nearest_e6m2(raw_scale)

    best_loss: torch.Tensor | None = None
    best_base: torch.Tensor | None = None
    best_lv2: torch.Tensor | None = None
    best_lv3: torch.Tensor | None = None
    for offset in base_offsets:
        base = nominal_base if offset == 0 else _e6m2_neighbor(nominal_base, offset)
        loss, lv2_bit, lv3_bit = _solve_hierarchy(
            absolute, base, importance
        )
        if best_loss is None:
            best_loss = loss
            best_base = base
            best_lv2 = lv2_bit
            best_lv3 = lv3_bit
        else:
            improve = loss < best_loss
            best_loss = torch.where(improve, loss, best_loss)
            best_base = torch.where(improve, base, best_base)
            best_lv2 = torch.where(improve.unsqueeze(-1), lv2_bit, best_lv2)
            best_lv3 = torch.where(
                improve.unsqueeze(-1).unsqueeze(-1), lv3_bit, best_lv3
            )

    assert best_base is not None and best_lv2 is not None and best_lv3 is not None
    scale_lv2 = best_lv2.to(torch.float32).add_(1.0)
    scale_lv3 = best_lv3.to(torch.float32).add_(1.0)
    total_scale = scale_lv2.unsqueeze(-1) * scale_lv3
    denominator = best_base[..., None, None, None] * total_scale.unsqueeze(-1)
    mant = torch.round(absolute * (4.0 / denominator)).clamp_(0.0, 7.0)
    mant.mul_(0.25)
    sign = torch.where(mant == 0, torch.zeros_like(restored), torch.sign(restored))
    return {
        "scale_factor": best_base.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
        "scale_lv2": scale_lv2.unsqueeze(-1).unsqueeze(-1),
        "scale_lv3": scale_lv3.unsqueeze(-1),
        "sign": sign,
        "mant": mant,
    }


def _channel_second_moment(
    calibration: list, channels: int, device: torch.device
) -> torch.Tensor:
    total = torch.zeros(channels, dtype=torch.float32, device=device)
    count = 0
    for item in calibration:
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            continue
        values = dequantize_nvfp4(item[0], item[1]).to(torch.float32)
        total.add_(values.square().reshape(-1, channels).sum(dim=0))
        count += values.numel() // channels
    if count == 0:
        total.fill_(1.0)
    else:
        total.div_(float(count))
    # Normalize inside each HiF4 block.  Clipping reduces calibration outlier
    # leverage without destroying persistent channel sensitivity.
    blocked = total.reshape(-1, HIF4_BLOCK_SIZE)
    blocked.div_(blocked.mean(dim=-1, keepdim=True).clamp_min_(1.0e-20))
    blocked.clamp_(_IMPORTANCE_MIN, _IMPORTANCE_MAX)
    blocked.pow_(_WEIGHT_IMPORTANCE_POWER)
    return blocked.reshape(-1, 8, 2, 4)


def _dequantize_hif4_params(params: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return (
        params["sign"]
        * params["mant"]
        * params["scale_lv3"]
        * params["scale_lv2"]
        * params["scale_factor"]
    ).flatten(-4)


def _converted_weight_column_energy(weight_params: HiF4Params) -> torch.Tensor:
    converted = _dequantize_hif4_params(weight_params).to(torch.float32)
    channels = int(converted.shape[-1])
    energy = converted.square().reshape(-1, channels).sum(dim=0)
    blocked = energy.reshape(-1, HIF4_BLOCK_SIZE)
    blocked.div_(blocked.mean(dim=-1, keepdim=True).clamp_min_(1.0e-20))
    blocked.clamp_(_IMPORTANCE_MIN, _IMPORTANCE_MAX)
    blocked.pow_(_ACTIVATION_IMPORTANCE_POWER)
    return blocked.reshape(-1, 8, 2, 4)


@torch.inference_mode()
def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    channels = int(weight_quant.shape[-1])
    activation_moment = _channel_second_moment(
        calib_activation_list, channels, weight_quant.device
    )
    weight_params = _quantize_hif4(
        weight_quant,
        weight_scale,
        importance=activation_moment if _USE_WEIGHT_IMPORTANCE else None,
        base_offsets=_WEIGHT_BASE_OFFSETS,
    )
    column_energy = _converted_weight_column_energy(weight_params)
    return {
        "weight_params": weight_params,
        "activation_state": {
            "version": 3,
            "column_energy": column_energy,
        },
    }


@torch.inference_mode()
def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_state: Any,
) -> HiF4Params:
    importance = None
    if _USE_ACTIVATION_IMPORTANCE and isinstance(activation_state, Mapping):
        importance = activation_state.get("column_energy")
    return _quantize_hif4(
        activation_quant,
        activation_scale,
        importance=importance,
        base_offsets=_ACTIVATION_BASE_OFFSETS,
    )


# Attention remains the exact v1 path in this Linear-only candidate.
@torch.inference_mode()
def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    del calib_qkv_list, q_num_heads, kv_num_heads, head_dim
    return {
        "q_state": {"version": 2},
        "k_state": {"version": 2},
        "v_state": {"version": 2},
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
