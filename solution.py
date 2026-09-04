"""
HiF4 solution.py 提交接口模板
==============================

本文件只说明参赛者需要实现的 6 个公开函数，以及这些函数的输入/输出数据契约。
不包含任何 HiF4 参考量化或标准量化实现。

必须实现的 6 个函数：

Linear:
    1. hif4_calibration_and_quantize_weight
    2. hif4_dynamic_quantize_activation

Attention:
    3. hif4_calibration_attention
    4. hif4_dynamic_quantize_q
    5. hif4_dynamic_quantize_k
    6. hif4_dynamic_quantize_v

说明：
- 输入的 Weight / Activation / Q / K / V 均以 NVFP4 carrier + block scale 的形式提供。
- calibration 函数可以根据校准数据生成后续动态量化需要使用的 state。
- dynamic 函数通过对应 state 获取 calibration 阶段产生的固定信息。
- 选手自行实现 HiF4 量化算法；本模板不会提供任何 HiF4 参考实现。
"""

from __future__ import annotations

from typing import Any

import torch


_HIF4_MIN_SCALE = 2.0 ** (-48)
_HIF4_MAX_SCALE = 49152.0
_HIF4_MAX_LOCAL_SCALE = 4.0
_HIF4_MAX_MANTISSA = 1.75
_HIF4_MAX_VALUE = _HIF4_MAX_SCALE * _HIF4_MAX_LOCAL_SCALE * _HIF4_MAX_MANTISSA
_HIF4_BLOCK_SIZE = 64
_NVFP4_BLOCK_SIZE = 16
_SEARCH_CHUNK_BLOCKS = 1024
_LINEAR_SMOOTH_ALPHA = 0.65
_ATTENTION_SMOOTH_ALPHA = 0.25
_SMOOTH_SCALE_MIN = 1.0 / 16.0
_SMOOTH_SCALE_MAX = 16.0

# The first candidate wins exact ties.  Keep this tuple ordered and stable so
# that repeated runs produce identical output tensors.
_GLOBAL_SCALE_MULTIPLIERS = (
    0.25,
    0.35,
    0.50,
    0.70,
    0.85,
    1.00,
    1.20,
    1.50,
    2.00,
    2.50,
    3.00,
    4.00,
)

# A second, block-local pass around the first-stage winner.  The baseline
# candidate (1.0) is always present, and a refined choice is materialized only
# after a meaningful local weighted-error reduction.
_REFINEMENT_SCALE_MULTIPLIERS = (0.75, 0.875, 1.0, 1.125, 1.25)
_MIN_REFINEMENT_RELATIVE_IMPROVEMENT = 0.01


# =============================================================================
# NVFP4 helper
# =============================================================================

def dequantize_nvfp4(
    quant_float: torch.Tensor,
    scale_float: torch.Tensor,
    blk_size: int = 16,
) -> torch.Tensor:
    """将接口中的 NVFP4 carrier 还原为 BF16 Tensor。

    Args:
        quant_float:
            NVFP4 value carrier，shape 为 ``(..., C)``。

        scale_float:
            NVFP4 block scale，shape 为 ``(..., C // blk_size)``。

        blk_size:
            NVFP4 block size，默认为 16。

    Returns:
        BF16 Tensor，shape 与 ``quant_float`` 相同。
    """
    return _dequantize_nvfp4_fp32(quant_float, scale_float, blk_size).to(
        torch.bfloat16
    )


def _dequantize_nvfp4_fp32(
    quant_float: torch.Tensor,
    scale_float: torch.Tensor,
    blk_size: int = _NVFP4_BLOCK_SIZE,
) -> torch.Tensor:
    """Dequantize the public NVFP4 carrier representation in FP32.

    The public helper intentionally returns BF16 for compatibility with the
    template.  The quantizer itself keeps the intermediate in FP32 so that the
    second quantization step does not add an avoidable BF16 rounding error.
    """

    if not isinstance(quant_float, torch.Tensor):
        raise TypeError("quant_float must be a torch.Tensor")
    if not isinstance(scale_float, torch.Tensor):
        raise TypeError("scale_float must be a torch.Tensor")
    if quant_float.ndim < 1:
        raise ValueError("quant_float must have at least one dimension")

    channels = int(quant_float.shape[-1])
    if channels % blk_size != 0:
        raise ValueError(
            f"last dimension {channels} is not divisible by block size {blk_size}"
        )

    expected_scale_shape = tuple(quant_float.shape[:-1]) + (channels // blk_size,)
    if tuple(scale_float.shape) != expected_scale_shape:
        raise ValueError(
            f"scale shape {tuple(scale_float.shape)} does not match "
            f"expected {expected_scale_shape}"
        )

    device = quant_float.device
    quant_fp32 = quant_float.detach().to(device=device, dtype=torch.float32)
    scale_fp32 = scale_float.detach().to(device=device, dtype=torch.float32)
    x = quant_fp32.unflatten(-1, (-1, blk_size))
    x = x * scale_fp32.unsqueeze(-1)
    return x.flatten(-2, -1)


def _snap_to_e6m2(value: torch.Tensor) -> torch.Tensor:
    """Snap positive FP32 values to the exact E6M2 grid used by self_check."""

    value = torch.nan_to_num(
        value,
        nan=_HIF4_MIN_SCALE,
        posinf=_HIF4_MAX_SCALE,
        neginf=_HIF4_MIN_SCALE,
    ).clamp(min=_HIF4_MIN_SCALE, max=_HIF4_MAX_SCALE)

    exponent = torch.floor(torch.log2(value))
    quantum = torch.pow(
        torch.tensor(2.0, dtype=value.dtype, device=value.device),
        exponent - 2.0,
    )
    snapped = torch.round(value / quantum) * quantum
    return snapped.clamp(min=_HIF4_MIN_SCALE, max=_HIF4_MAX_SCALE)


def _local_scale_options(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return all 8 legal (lv2, lv3-left, lv3-right) combinations."""

    lv2 = torch.tensor(
        [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
        dtype=torch.float32,
        device=device,
    )
    lv3 = torch.tensor(
        [
            [1.0, 1.0],
            [1.0, 2.0],
            [2.0, 1.0],
            [2.0, 2.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [2.0, 1.0],
            [2.0, 2.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    return lv2, lv3, lv2.unsqueeze(-1) * lv3


def _evaluate_hif4_scale_candidates(
    abs_block: torch.Tensor,
    candidate_scales: torch.Tensor,
    local_scale_options: torch.Tensor,
    weight_block: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return total errors and local exponent choices for global scales."""

    # values       [block, candidate, lv2-group, option, lv3-group, value]
    # local scales [1,     1,         1,         option, lv3-group, 1]
    values = abs_block[:, None, :, None, :, :]
    global_scales = candidate_scales[:, :, None, None, None, None]
    local_scales = local_scale_options[None, None, None, :, :, None]
    total_scales = global_scales * local_scales

    candidate_mant = torch.round(values / total_scales * 4.0) * 0.25
    candidate_mant = candidate_mant.clamp(
        min=0.0,
        max=_HIF4_MAX_MANTISSA,
    )
    reconstructed = candidate_mant * total_scales
    element_error = (values - reconstructed).square()
    if weight_block is not None:
        element_error = element_error * weight_block[:, None, :, None, :, :]
    squared_error = element_error.sum(dim=(-1, -2))

    # Each 8-value group independently chooses one of the 8 legal local
    # exponent combinations.  Their errors then sum for the 64-value block.
    local_error, local_choice = squared_error.min(dim=-1)
    return local_error.sum(dim=-1), local_choice


def _quantize_hif4(
    x: torch.Tensor,
    error_weights: torch.Tensor | None = None,
    enable_refinement: bool = True,
) -> dict[str, torch.Tensor]:
    """Quantize FP32 values with guarded two-stage weighted MSE search."""

    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.ndim < 1:
        raise ValueError("x must have at least one dimension")
    if type(enable_refinement) is not bool:
        raise TypeError("enable_refinement must be a bool")

    channels = int(x.shape[-1])
    if channels % _HIF4_BLOCK_SIZE != 0:
        raise ValueError(
            f"last dimension {channels} is not divisible by HiF4 block size "
            f"{_HIF4_BLOCK_SIZE}"
        )

    x_fp32 = x.detach().to(dtype=torch.float32)
    x_fp32 = torch.nan_to_num(
        x_fp32,
        nan=0.0,
        posinf=_HIF4_MAX_VALUE,
        neginf=-_HIF4_MAX_VALUE,
    ).clamp(min=-_HIF4_MAX_VALUE, max=_HIF4_MAX_VALUE)

    weight_blocks: torch.Tensor | None = None
    if error_weights is not None:
        if not isinstance(error_weights, torch.Tensor):
            raise TypeError("error_weights must be a torch.Tensor or None")
        weights = error_weights.detach().to(device=x_fp32.device, dtype=torch.float32)
        try:
            weights = torch.broadcast_to(weights, x_fp32.shape)
        except RuntimeError as exc:
            raise ValueError(
                f"error_weights shape {tuple(error_weights.shape)} cannot broadcast "
                f"to input shape {tuple(x_fp32.shape)}"
            ) from exc
        if (weights < 0.0).any():
            raise ValueError("error_weights must be non-negative")
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights = weights / weights.mean().clamp_min(1.0e-12)
        weight_blocks = weights.reshape(-1, 8, 2, 4)

    prefix = tuple(int(size) for size in x_fp32.shape[:-1])
    block_count_per_row = channels // _HIF4_BLOCK_SIZE
    blocks = x_fp32.reshape(-1, 8, 2, 4)
    total_blocks = int(blocks.shape[0])
    device = x_fp32.device

    scale_factor_out = torch.empty(
        (total_blocks, 1, 1, 1), dtype=torch.float32, device=device
    )
    scale_lv2_out = torch.empty(
        (total_blocks, 8, 1, 1), dtype=torch.float32, device=device
    )
    scale_lv3_out = torch.empty(
        (total_blocks, 8, 2, 1), dtype=torch.float32, device=device
    )
    sign_out = torch.empty(
        (total_blocks, 8, 2, 4), dtype=torch.float32, device=device
    )
    mant_out = torch.empty_like(sign_out)

    multipliers = torch.tensor(
        _GLOBAL_SCALE_MULTIPLIERS,
        dtype=torch.float32,
        device=device,
    )
    refinement_multipliers = torch.tensor(
        _REFINEMENT_SCALE_MULTIPLIERS,
        dtype=torch.float32,
        device=device,
    )
    lv2_options, lv3_options, local_scale_options = _local_scale_options(device)

    for start in range(0, total_blocks, _SEARCH_CHUNK_BLOCKS):
        end = min(start + _SEARCH_CHUNK_BLOCKS, total_blocks)
        block = blocks[start:end]
        abs_block = block.abs()

        # The largest representable local factor is 2 * 2 * 1.75 = 7.
        base_scale = abs_block.amax(dim=(1, 2, 3)) / 7.0
        candidate_scales = _snap_to_e6m2(
            base_scale.unsqueeze(-1) * multipliers.unsqueeze(0)
        )
        weight_block = None if weight_blocks is None else weight_blocks[start:end]
        global_error, local_choice = _evaluate_hif4_scale_candidates(
            abs_block,
            candidate_scales,
            local_scale_options,
            weight_block,
        )
        global_choice = global_error.argmin(dim=-1)

        row_index = torch.arange(end - start, device=device)
        baseline_scale = candidate_scales[row_index, global_choice]
        baseline_error = global_error[row_index, global_choice]
        baseline_local = local_choice.gather(
            1,
            global_choice[:, None, None].expand(-1, 1, 8),
        ).squeeze(1)
        chosen_scale = baseline_scale
        chosen_local = baseline_local

        if enable_refinement:
            refined_scales = _snap_to_e6m2(
                baseline_scale.unsqueeze(-1)
                * refinement_multipliers.unsqueeze(0)
            )
            refined_error, refined_local_choice = (
                _evaluate_hif4_scale_candidates(
                    abs_block,
                    refined_scales,
                    local_scale_options,
                    weight_block,
                )
            )
            refined_global_choice = refined_error.argmin(dim=-1)
            best_refined_error = refined_error[
                row_index,
                refined_global_choice,
            ]
            best_refined_scale = refined_scales[
                row_index,
                refined_global_choice,
            ]
            best_refined_local = refined_local_choice.gather(
                1,
                refined_global_choice[:, None, None].expand(-1, 1, 8),
            ).squeeze(1)
            use_refinement = best_refined_error < baseline_error * (
                1.0 - _MIN_REFINEMENT_RELATIVE_IMPROVEMENT
            )
            chosen_scale = torch.where(
                use_refinement,
                best_refined_scale,
                baseline_scale,
            )
            chosen_local = torch.where(
                use_refinement[:, None],
                best_refined_local,
                baseline_local,
            )

        chosen_lv2 = lv2_options[chosen_local]
        chosen_lv3 = lv3_options[chosen_local]
        chosen_total_scale = (
            chosen_scale[:, None, None, None]
            * chosen_lv2[:, :, None, None]
            * chosen_lv3[:, :, :, None]
        )

        chosen_mant = torch.round(abs_block / chosen_total_scale * 4.0) * 0.25
        chosen_mant = chosen_mant.clamp(min=0.0, max=_HIF4_MAX_MANTISSA)
        chosen_sign = torch.sign(block)
        chosen_sign = torch.where(
            chosen_mant == 0.0,
            torch.zeros((), dtype=torch.float32, device=device),
            chosen_sign,
        )

        scale_factor_out[start:end, 0, 0, 0] = chosen_scale
        scale_lv2_out[start:end, :, 0, 0] = chosen_lv2
        scale_lv3_out[start:end, :, :, 0] = chosen_lv3
        sign_out[start:end] = chosen_sign
        mant_out[start:end] = chosen_mant

    return {
        "scale_factor": scale_factor_out.reshape(
            prefix + (block_count_per_row, 1, 1, 1)
        ).contiguous(),
        "scale_lv2": scale_lv2_out.reshape(
            prefix + (block_count_per_row, 8, 1, 1)
        ).contiguous(),
        "scale_lv3": scale_lv3_out.reshape(
            prefix + (block_count_per_row, 8, 2, 1)
        ).contiguous(),
        "sign": sign_out.reshape(
            prefix + (block_count_per_row, 8, 2, 4)
        ).contiguous(),
        "mant": mant_out.reshape(
            prefix + (block_count_per_row, 8, 2, 4)
        ).contiguous(),
    }


def _quantize_nvfp4_pair(
    quant_float: torch.Tensor,
    scale_float: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return _quantize_hif4(
        _dequantize_nvfp4_fp32(quant_float, scale_float, _NVFP4_BLOCK_SIZE)
    )


def _block_signs64(block_count: int, device: torch.device) -> torch.Tensor:
    """Return deterministic Rademacher signs, one vector per 64D block."""

    block = torch.arange(block_count, dtype=torch.int64, device=device)[:, None]
    lane = torch.arange(64, dtype=torch.int64, device=device)[None, :]
    hashed = (lane * 1103515245 + block * 12345 + 0x9E3779B9) & 0x7FFFFFFF
    return torch.where(((hashed >> 16) & 1) == 0, 1.0, -1.0)


def _apply_block_hadamard(x: torch.Tensor) -> torch.Tensor:
    """Apply a fixed signed orthonormal Hadamard transform per HiF4 block."""

    if int(x.shape[-1]) % _HIF4_BLOCK_SIZE != 0:
        raise ValueError("Hadamard input channels must be divisible by 64")
    original_shape = tuple(x.shape)
    y = x.to(dtype=torch.float32).reshape(-1, x.shape[-1] // 64, 64)
    y = y * _block_signs64(int(y.shape[1]), y.device).unsqueeze(0)
    for width in (1, 2, 4, 8, 16, 32):
        groups = y.reshape(*y.shape[:-1], -1, 2 * width)
        left = groups[..., :width]
        right = groups[..., width:]
        y = torch.cat((left + right, left - right), dim=-1).reshape(*y.shape)
    return (y * 0.125).reshape(original_shape)


def _apply_attention_hadamard(
    x: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Apply identical block transforms inside every attention head."""

    original_shape = tuple(x.shape)
    per_head = x.reshape(-1, num_heads, head_dim).reshape(-1, head_dim)
    return _apply_block_hadamard(per_head).reshape(original_shape)


def _smooth_scale(
    left_abs_max: torch.Tensor,
    right_abs_max: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Construct a bounded reciprocal equivalent-transformation scale."""

    left = left_abs_max.float().clamp_min(1.0e-6)
    right = right_abs_max.float().clamp_min(1.0e-6)
    scale = left.pow(alpha) / right.pow(1.0 - alpha)
    return torch.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=1.0).clamp(
        min=_SMOOTH_SCALE_MIN,
        max=_SMOOTH_SCALE_MAX,
    )


def _state_tensor(state: Any, key: str, expected_shape: tuple[int, ...]) -> torch.Tensor:
    if type(state) is not dict or type(state.get(key)) is not torch.Tensor:
        raise ValueError(f"state must contain CPU tensor {key!r}")
    value = state[key]
    if tuple(value.shape) != expected_shape:
        raise ValueError(
            f"state[{key!r}] shape {tuple(value.shape)} != {expected_shape}"
        )
    return value.detach().to(dtype=torch.float32)


def _make_state(role: str) -> dict[str, Any]:
    """Create a small self_check-compatible immutable-by-convention state."""

    return {
        "schema_version": 5,
        "algorithm": "hif4-v4-guarded-e6m2-refinement",
        "role": role,
        "calibration_used": True,
    }


def _validate_attention_shape(
    quant_float: torch.Tensor,
    num_heads: int,
    head_dim: int,
    role: str,
) -> None:
    if type(num_heads) is not int or num_heads <= 0:
        raise ValueError(f"{role}: num_heads must be a positive int")
    if type(head_dim) is not int or head_dim <= 0:
        raise ValueError(f"{role}: head_dim must be a positive int")
    if quant_float.ndim != 2:
        raise ValueError(f"{role}: quant tensor must be 2D [seq_len, hidden]")
    expected_hidden = num_heads * head_dim
    if int(quant_float.shape[-1]) != expected_hidden:
        raise ValueError(
            f"{role}: hidden size {quant_float.shape[-1]} does not match "
            f"num_heads * head_dim ({expected_hidden})"
        )


# =============================================================================
# 返回值公共说明
# =============================================================================
#
# HiF4Params
# -----------
# 所有需要返回 HiF4 量化结果的函数，都应返回一个 dict，并至少包含以下 5 个
# torch.Tensor：
#
#     {
#         "scale_factor": ...,
#         "scale_lv2":    ...,
#         "scale_lv3":    ...,
#         "sign":         ...,
#         "mant":         ...,
#     }
#
# 若原 Tensor shape 为 ``(*prefix, C)``，其中 C % 64 == 0，则五个字段的 shape 为：
#
#     scale_factor : (*prefix, C // 64, 1, 1, 1)
#     scale_lv2    : (*prefix, C // 64, 8, 1, 1)
#     scale_lv3    : (*prefix, C // 64, 8, 2, 1)
#     sign         : (*prefix, C // 64, 8, 2, 4)
#     mant         : (*prefix, C // 64, 8, 2, 4)
#
# 数值格式要求：
#
#     scale_factor : HiF4 E6M2 scale
#     scale_lv2    : 1 或 2
#     scale_lv3    : 1 或 2
#     sign         : -1、0 或 1
#     mant         : 0 ~ 1.75，步长 0.25
#
# 对应反量化关系为：
#
#     x_hat = sign * mant * scale_lv3 * scale_lv2 * scale_factor
#
# -----------------------------------------------------------------------------
# State
# -----------------------------------------------------------------------------
# calibration 函数返回的 activation_state / q_state / k_state / v_state 用于把
# calibration 阶段得到的信息传给对应的 dynamic quantization 函数。
#
# 推荐使用纯数据结构，例如：
#
#     None / bool / int / finite float / str
#     CPU torch.Tensor
#     list / tuple
#     dict[str, ...]
#
# 不要依赖自定义 Python 对象、可调用对象或外部可变状态。
#
# Linear 的 activation_state 可以包含固定 Weight 相关信息；例如选手可以根据自己的
# 算法保存 smooth scale、clip 参数、importance，或固定的 weight quantization 参数。
#


# =============================================================================
# 1. Linear calibration + Weight quantization
# =============================================================================

def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    """使用 Weight 和 calibration Activation 完成离线校准，并量化 Weight。

    Args:
        weight_quant:
            Weight 的 NVFP4 value carrier。
            shape 通常为 ``[out_features, in_features]``。

        weight_scale:
            Weight 的 NVFP4 block scale。
            若 ``weight_quant.shape == [M, K]``，则通常为
            ``[M, K // 16]``。

        calib_activation_list:
            当前 Weight 对应的 calibration Activation 列表。

            每个元素均为一个二元 NVFP4 pair：

                (activation_quant, activation_scale)

            其中：

                activation_quant : [tokens, in_features]
                activation_scale : [tokens, in_features // 16]

            calibration 阶段可以同时利用 Weight 和这些 Activation 搜索：
            smooth scale、clip 参数、旋转参数、importance、量化参数等。

    Returns:
        必须返回：

            {
                "weight_params": HiF4Params,
                "activation_state": state,
            }

        weight_params:
            当前 Weight 的最终 HiF4 参数。
            它对应 ``weight_quant`` 解码后的原始 Weight shape。

        activation_state:
            传给 ``hif4_dynamic_quantize_activation`` 的 calibration state。

            这里可以保存后续在线 Activation 量化所需的固定信息，例如：
                - smooth scale
                - clip/search 参数
                - channel importance
                - rotation 参数
                - 固定 Weight 相关参数
                - 其他纯数据 calibration 结果

    Important:
        本函数必须自行实现 Weight 的 HiF4 量化算法。
        本模板不提供 HiF4 参考实现。
    """
    weight = _dequantize_nvfp4_fp32(weight_quant, weight_scale)
    if weight.ndim != 2:
        raise ValueError("weight must be a 2D tensor")
    if not isinstance(calib_activation_list, list) or not calib_activation_list:
        raise ValueError("calib_activation_list must be a non-empty list")

    channels = int(weight.shape[-1])
    activation_abs_max = torch.zeros(
        channels,
        dtype=torch.float32,
        device=weight.device,
    )
    for quant, scale in calib_activation_list:
        activation = _dequantize_nvfp4_fp32(quant, scale)
        if int(activation.shape[-1]) != channels:
            raise ValueError("calibration activation channels do not match weight")
        activation_abs_max = torch.maximum(
            activation_abs_max,
            activation.abs().amax(dim=tuple(range(activation.ndim - 1))),
        )

    smooth = _smooth_scale(
        activation_abs_max,
        weight.abs().amax(dim=0),
        _LINEAR_SMOOTH_ALPHA,
    )
    transformed_weight = _apply_block_hadamard(weight * smooth)

    activation_second = torch.zeros_like(smooth)
    activation_count = 0
    for quant, scale in calib_activation_list:
        activation = _dequantize_nvfp4_fp32(quant, scale)
        transformed = _apply_block_hadamard(activation / smooth)
        activation_second += transformed.square().sum(
            dim=tuple(range(transformed.ndim - 1))
        )
        activation_count += transformed.numel() // channels
    activation_second = activation_second / max(activation_count, 1)
    activation_second = activation_second.clamp_min(1.0e-8)

    activation_importance = transformed_weight.square().sum(dim=0).clamp_min(1.0e-8)
    state = _make_state("activation")
    state.update({
        "smooth_scale": smooth.detach().cpu().contiguous(),
        "error_weights": activation_importance.detach().cpu().contiguous(),
        "smooth_alpha": _LINEAR_SMOOTH_ALPHA,
        "rotation": "signed-hadamard-64",
    })
    return {
        "weight_params": _quantize_hif4(transformed_weight, activation_second),
        "activation_state": state,
    }


# =============================================================================
# 2. Dynamic Activation quantization
# =============================================================================

def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_state: Any,
) -> dict[str, torch.Tensor]:
    """对当前 Activation 动态生成 HiF4 参数。

    Args:
        activation_quant:
            当前 Activation 的 NVFP4 value carrier，通常为
            ``[tokens, hidden_size]``。

        activation_scale:
            当前 Activation 的 NVFP4 block scale，通常为
            ``[tokens, hidden_size // 16]``。

        activation_state:
            与当前 Linear Weight 对应，由
            ``hif4_calibration_and_quantize_weight`` 返回的 state。

            dynamic quantization 可以使用这里保存的 calibration 信息和固定
            Weight 相关信息，再结合当前 Activation 自身进行搜索或动态决策。

    Returns:
        当前 Activation 对应的 HiF4Params。
        输出参数的逻辑 Tensor shape 必须与当前 Activation 一致。
    """
    channels = int(activation_quant.shape[-1])
    smooth = _state_tensor(activation_state, "smooth_scale", (channels,))
    error_weights = _state_tensor(
        activation_state,
        "error_weights",
        (channels,),
    )
    activation = _dequantize_nvfp4_fp32(activation_quant, activation_scale)
    transformed = _apply_block_hadamard(activation / smooth)
    return _quantize_hif4(transformed, error_weights)


# =============================================================================
# 3. Attention calibration
# =============================================================================

def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    """使用 calibration Q/K/V 生成 Q、K、V 后续动态量化所需的 state。

    Args:
        calib_qkv_list:
            calibration Q/K/V sample 列表。

            每个 sample 的标准结构为：

                {
                    "q": (q_quant, q_scale),
                    "k": (k_quant, k_scale),
                    "v": (v_quant, v_scale),
                }

            其中 quant Tensor 均为二维：

                q_quant : [seq_len, q_num_heads  * head_dim]
                k_quant : [seq_len, kv_num_heads * head_dim]
                v_quant : [seq_len, kv_num_heads * head_dim]

            对应 scale Tensor 的最后一维为 quant Tensor 最后一维 / 16。

        q_num_heads:
            Query head 数。

        kv_num_heads:
            Key / Value head 数。

        head_dim:
            每个 attention head 的维度。

    Returns:
        必须返回：

            {
                "q_state": q_state,
                "k_state": k_state,
                "v_state": v_state,
            }

        q_state:
            传给 ``hif4_dynamic_quantize_q``。

        k_state:
            传给 ``hif4_dynamic_quantize_k``。

        v_state:
            传给 ``hif4_dynamic_quantize_v``。

        三个 state 可以保存 calibration 阶段得到的固定纯数据参数，例如：
            - clip 参数
            - per-head / per-channel scale
            - rotation 参数
            - importance
            - 其他动态量化需要使用的 calibration 结果
    """
    if not isinstance(calib_qkv_list, list) or not calib_qkv_list:
        raise ValueError("calib_qkv_list must be a non-empty list")
    if type(q_num_heads) is not int or q_num_heads <= 0:
        raise ValueError("q_num_heads must be a positive int")
    if type(kv_num_heads) is not int or kv_num_heads <= 0:
        raise ValueError("kv_num_heads must be a positive int")
    if type(head_dim) is not int or head_dim <= 0:
        raise ValueError("head_dim must be a positive int")
    if q_num_heads % kv_num_heads != 0:
        raise ValueError("q_num_heads must be divisible by kv_num_heads")

    q_abs_max = torch.zeros((q_num_heads, head_dim), dtype=torch.float32)
    k_abs_max = torch.zeros((kv_num_heads, head_dim), dtype=torch.float32)
    for sample in calib_qkv_list:
        if type(sample) is not dict:
            raise ValueError("each attention calibration sample must be a dict")
        q_quant, q_scale = sample["q"]
        k_quant, k_scale = sample["k"]
        q = _dequantize_nvfp4_fp32(q_quant, q_scale)
        k = _dequantize_nvfp4_fp32(k_quant, k_scale)
        _validate_attention_shape(q_quant, q_num_heads, head_dim, "q calibration")
        _validate_attention_shape(k_quant, kv_num_heads, head_dim, "k calibration")
        q_abs_max = torch.maximum(
            q_abs_max,
            q.reshape(-1, q_num_heads, head_dim).abs().amax(dim=0).cpu(),
        )
        k_abs_max = torch.maximum(
            k_abs_max,
            k.reshape(-1, kv_num_heads, head_dim).abs().amax(dim=0).cpu(),
        )

    q_per_kv = q_abs_max.reshape(
        kv_num_heads,
        q_num_heads // kv_num_heads,
        head_dim,
    ).amax(dim=1)
    kv_smooth = _smooth_scale(
        q_per_kv,
        k_abs_max,
        _ATTENTION_SMOOTH_ALPHA,
    ).cpu()
    q_smooth = kv_smooth[:, None, :].expand(
        kv_num_heads,
        q_num_heads // kv_num_heads,
        head_dim,
    ).reshape(q_num_heads, head_dim).contiguous()

    q_state = _make_state("q")
    q_state.update({
        "smooth_scale": q_smooth,
        "smooth_alpha": _ATTENTION_SMOOTH_ALPHA,
        "rotation": "per-head-signed-hadamard-64",
    })
    k_state = _make_state("k")
    k_state.update({
        "smooth_scale": kv_smooth.contiguous(),
        "smooth_alpha": _ATTENTION_SMOOTH_ALPHA,
        "rotation": "per-head-signed-hadamard-64",
    })
    v_state = _make_state("v")
    v_state["calibration_used"] = False
    v_state["rotation"] = "none"
    return {"q_state": q_state, "k_state": k_state, "v_state": v_state}


# =============================================================================
# 4. Dynamic Q quantization
# =============================================================================

def hif4_dynamic_quantize_q(
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    q_num_heads: int,
    head_dim: int,
    q_state: Any,
) -> dict[str, torch.Tensor]:
    """对当前 Query Tensor 动态生成 HiF4 参数。

    Args:
        q_quant:
            Q 的 NVFP4 value carrier，shape 为
            ``[seq_len, q_num_heads * head_dim]``。

        q_scale:
            Q 的 NVFP4 block scale，shape 为
            ``[seq_len, q_num_heads * head_dim // 16]``。

        q_num_heads:
            Query head 数。

        head_dim:
            每个 Query head 的维度。

        q_state:
            ``hif4_calibration_attention`` 返回的 Q calibration state。

    Returns:
        当前 Q 对应的 HiF4Params。
    """
    _validate_attention_shape(q_quant, q_num_heads, head_dim, "q")
    smooth = _state_tensor(
        q_state,
        "smooth_scale",
        (q_num_heads, head_dim),
    )
    q = _dequantize_nvfp4_fp32(q_quant, q_scale)
    transformed = (q.reshape(-1, q_num_heads, head_dim) / smooth).reshape_as(q)
    transformed = _apply_attention_hadamard(
        transformed,
        q_num_heads,
        head_dim,
    )
    return _quantize_hif4(transformed)


# =============================================================================
# 5. Dynamic K quantization
# =============================================================================

def hif4_dynamic_quantize_k(
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    k_state: Any,
) -> dict[str, torch.Tensor]:
    """对当前 Key Tensor 动态生成 HiF4 参数。

    Args:
        k_quant:
            K 的 NVFP4 value carrier，shape 为
            ``[seq_len, kv_num_heads * head_dim]``。

        k_scale:
            K 的 NVFP4 block scale，shape 为
            ``[seq_len, kv_num_heads * head_dim // 16]``。

        kv_num_heads:
            Key / Value head 数。

        head_dim:
            每个 Key head 的维度。

        k_state:
            ``hif4_calibration_attention`` 返回的 K calibration state。

    Returns:
        当前 K 对应的 HiF4Params。
    """
    _validate_attention_shape(k_quant, kv_num_heads, head_dim, "k")
    smooth = _state_tensor(
        k_state,
        "smooth_scale",
        (kv_num_heads, head_dim),
    )
    k = _dequantize_nvfp4_fp32(k_quant, k_scale)
    transformed = (k.reshape(-1, kv_num_heads, head_dim) * smooth).reshape_as(k)
    transformed = _apply_attention_hadamard(
        transformed,
        kv_num_heads,
        head_dim,
    )
    return _quantize_hif4(transformed)


# =============================================================================
# 6. Dynamic V quantization
# =============================================================================

def hif4_dynamic_quantize_v(
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    v_state: Any,
) -> dict[str, torch.Tensor]:
    """对当前 Value Tensor 动态生成 HiF4 参数。

    Args:
        v_quant:
            V 的 NVFP4 value carrier，shape 为
            ``[seq_len, kv_num_heads * head_dim]``。

        v_scale:
            V 的 NVFP4 block scale，shape 为
            ``[seq_len, kv_num_heads * head_dim // 16]``。

        kv_num_heads:
            Key / Value head 数。

        head_dim:
            每个 Value head 的维度。

        v_state:
            ``hif4_calibration_attention`` 返回的 V calibration state。

    Returns:
        当前 V 对应的 HiF4Params。
    """
    _ = v_state
    _validate_attention_shape(v_quant, kv_num_heads, head_dim, "v")
    return _quantize_nvfp4_pair(v_quant, v_scale)
