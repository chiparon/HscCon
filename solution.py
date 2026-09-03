"""Standalone matrix- and Attention-adaptive HiF4 v2 candidate.

The Linear path applies a diagonal Smooth transform followed by the same
orthonormal Walsh-Hadamard transform to every 64 input channels.  For row
vectors, ``A' = (A / s) H`` and ``W' = (W * s) H``; therefore the floating
point product is unchanged because ``H H.T = I``.  The default uses the root
exact hierarchy DP; optional diagonal weighting remains only as an ablation
knob and never forms ``A @ W.T``.

Attention uses a reciprocal Smooth-QK transform followed by the same H64.
A calibration-only structural/stability gate limits it to sufficiently grouped,
repeatable GQA; rejected layouts fall back to the exact fixed-base path.
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

# Experiment knobs.  The benchmark changes these before calibration.
_USE_SMOOTH = True
_USE_HADAMARD = True
_SMOOTH_ALPHA = 0.65
_SMOOTH_LIMIT = 16.0
_SMOOTH_STAT = "rms"
_SMOOTH_BLOCK_SHRINK = 0.5
_USE_WEIGHT_IMPORTANCE = True
_USE_ACTIVATION_IMPORTANCE = True
_IMPORTANCE_MIN = 0.01
_IMPORTANCE_MAX = 64.0
_IMPORTANCE_POWER = 0.5
_WEIGHT_BASE_OFFSETS = (0, 1)
_ACTIVATION_BASE_OFFSETS = (0, 1)
_USE_MULTIBASE_V = True
_USE_SPARSE_HESSIAN_WEIGHT_REFINEMENT = True
_SPARSE_HESSIAN_TOP_FRACTION = 0.02
_SPARSE_HESSIAN_CALIBRATION_ROWS = 8
_SPARSE_HESSIAN_DAMPING = 0.01
_SPARSE_HESSIAN_MIN_IMPROVEMENT = 0.01
_SPARSE_HESSIAN_CHUNK_BLOCKS = 4096
_SPARSE_HESSIAN_MIN_CALIBRATION_SAMPLES = 5
_SPARSE_HESSIAN_MIN_WEIGHT_ROWS = 4096
_SPARSE_HESSIAN_MIN_CHANNELS = 1024
_ATTENTION_MIN_GQA_RATIO = 4
_ATTENTION_MIN_CALIBRATION_SAMPLES = 2
_ATTENTION_MAX_LOG_SMOOTH_STD = 0.22
_ATTENTION_MAX_CROSS_SAMPLE_STD = 0.16
_ATTENTION_Q_BASE_OFFSETS = (0, -1, 1, -2, 2)
_ATTENTION_Q_MIN_IMPROVEMENT = 0.01
_ATTENTION_V_BASE_OFFSETS = (0, -1, 1, -2, 2)
_ATTENTION_V_MIN_IMPROVEMENT = 0.02


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


def _restore_dense(
    quant_float: torch.Tensor, scale_float: torch.Tensor
) -> torch.Tensor:
    return dequantize_nvfp4(quant_float, scale_float).to(torch.float32)


def _nearest_e6m2(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32).clamp(min=_E6M2_MIN, max=_E6M2_MAX).contiguous()
    bits = x.view(torch.int32)
    retained_lsb = (bits >> 21) & 1
    rounded_bits = (bits + 0x000FFFFF + retained_lsb) & ~0x001FFFFF
    return rounded_bits.view(torch.float32).clamp_(max=_E6M2_MAX)


def _e6m2_neighbor(base: torch.Tensor, offset: int) -> torch.Tensor:
    bits = base.contiguous().view(torch.int32)
    shifted = (bits + offset * _E6M2_STEP_BITS).clamp(
        _E6M2_MIN_BITS, _E6M2_MAX_BITS
    )
    return shifted.view(torch.float32)


def _hadamard64(values: torch.Tensor) -> torch.Tensor:
    """Apply a normalized Sylvester H64 on the final axis, block by block."""
    if values.shape[-1] % HIF4_BLOCK_SIZE:
        raise ValueError("Hadamard transform requires a multiple of 64 channels")
    shape = values.shape
    transformed = values.reshape(-1, HIF4_BLOCK_SIZE)
    width = 1
    while width < HIF4_BLOCK_SIZE:
        pairs = transformed.reshape(-1, HIF4_BLOCK_SIZE // (2 * width), 2, width)
        left = pairs[:, :, 0, :]
        right = pairs[:, :, 1, :]
        transformed = torch.cat((left + right, left - right), dim=-1).reshape(
            -1, HIF4_BLOCK_SIZE
        )
        width *= 2
    return transformed.reshape(shape).mul_(HIF4_BLOCK_SIZE**-0.5)


def _transform(values: torch.Tensor, smooth: torch.Tensor, inverse: bool) -> torch.Tensor:
    if _USE_SMOOTH:
        values = values / smooth if inverse else values * smooth
    if _USE_HADAMARD:
        values = _hadamard64(values)
    return values


def _prepare_importance(
    importance: torch.Tensor | None, blocked: torch.Tensor
) -> torch.Tensor | None:
    if importance is None:
        return None
    expected = blocked.shape[-4:]
    if importance.numel() != expected.numel():
        raise ValueError(
            f"importance has {importance.numel()} elements, expected {expected.numel()}"
        )
    return importance.to(device=blocked.device, dtype=torch.float32).reshape(expected)


def _loss_at_exponent(
    absolute: torch.Tensor,
    base_scale: torch.Tensor,
    exponent_scale: float,
    importance: torch.Tensor | None,
) -> torch.Tensor:
    denominator = base_scale[..., None, None, None] * exponent_scale
    residual = torch.round(absolute * (4.0 / denominator)).clamp_(0.0, 7.0)
    residual.mul_(denominator * 0.25).sub_(absolute).square_()
    if importance is not None:
        residual.mul_(importance)
    return residual.sum(dim=-1)


def _solve_hierarchy(
    absolute: torch.Tensor,
    base_scale: torch.Tensor,
    importance: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    loss0 = _loss_at_exponent(absolute, base_scale, 1.0, importance)
    loss1 = _loss_at_exponent(absolute, base_scale, 2.0, importance)
    loss2 = _loss_at_exponent(absolute, base_scale, 4.0, importance)
    path0 = torch.minimum(loss0, loss1).sum(dim=-1)
    path1 = torch.minimum(loss1, loss2).sum(dim=-1)
    lv2_bit = path1 < path0
    lv3_bit = torch.where(
        lv2_bit.unsqueeze(-1), loss2 < loss1, loss1 < loss0
    )
    best_loss = torch.where(lv2_bit, path1, path0).sum(dim=-1)
    return best_loss, lv2_bit, lv3_bit


def _quantize_dense(
    values: torch.Tensor,
    importance: torch.Tensor | None = None,
    base_offsets: tuple[int, ...] = (0,),
    min_relative_improvement: float = 0.0,
) -> HiF4Params:
    channels = int(values.shape[-1])
    if channels % HIF4_BLOCK_SIZE:
        raise ValueError(f"last dimension {channels} is not divisible by 64")
    prefix = values.shape[:-1]
    absolute = values.to(torch.float32).reshape(*prefix, -1, 8, 2, 4).abs()
    importance = _prepare_importance(importance, absolute)
    peak64 = absolute.amax(dim=(-1, -2, -3))
    raw_scale = (peak64.to(torch.bfloat16) * _BF16_ONE_OVER_SEVEN).float()
    nominal = _nearest_e6m2(raw_scale)

    best_loss = None
    best_base = None
    best_lv2 = None
    best_lv3 = None
    baseline_loss = None
    baseline_lv2 = None
    baseline_lv3 = None
    for offset in base_offsets:
        base = nominal if offset == 0 else _e6m2_neighbor(nominal, offset)
        loss, lv2_bit, lv3_bit = _solve_hierarchy(absolute, base, importance)
        if best_loss is None:
            best_loss, best_base = loss, base
            best_lv2, best_lv3 = lv2_bit, lv3_bit
            baseline_loss = loss
            baseline_lv2, baseline_lv3 = lv2_bit, lv3_bit
        else:
            improve = loss < best_loss
            best_loss = torch.where(improve, loss, best_loss)
            best_base = torch.where(improve, base, best_base)
            best_lv2 = torch.where(improve.unsqueeze(-1), lv2_bit, best_lv2)
            best_lv3 = torch.where(improve[:, None, None] if improve.ndim == 1 else improve.unsqueeze(-1).unsqueeze(-1), lv3_bit, best_lv3)

    assert best_base is not None and best_lv2 is not None and best_lv3 is not None
    if min_relative_improvement:
        assert baseline_loss is not None
        assert baseline_lv2 is not None and baseline_lv3 is not None
        accept = best_loss < baseline_loss * (1.0 - min_relative_improvement)
        best_base = torch.where(accept, best_base, nominal)
        best_lv2 = torch.where(accept.unsqueeze(-1), best_lv2, baseline_lv2)
        best_lv3 = torch.where(
            accept.unsqueeze(-1).unsqueeze(-1), best_lv3, baseline_lv3
        )
    scale_lv2 = best_lv2.float().add_(1.0)
    scale_lv3 = best_lv3.float().add_(1.0)
    total_scale = scale_lv2.unsqueeze(-1) * scale_lv3
    denominator = best_base[..., None, None, None] * total_scale.unsqueeze(-1)
    mant = torch.round(absolute * (4.0 / denominator)).clamp_(0.0, 7.0)
    mant.mul_(0.25)
    sign = torch.where(mant == 0, torch.zeros_like(values.reshape_as(absolute)), torch.sign(values.reshape_as(absolute)))
    return {
        "scale_factor": best_base.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
        "scale_lv2": scale_lv2.unsqueeze(-1).unsqueeze(-1),
        "scale_lv3": scale_lv3.unsqueeze(-1),
        "sign": sign,
        "mant": mant,
    }


def _importance_from_channel_values(values: torch.Tensor) -> torch.Tensor:
    blocked = values.float().reshape(-1, HIF4_BLOCK_SIZE)
    blocked.div_(blocked.mean(dim=-1, keepdim=True).clamp_min_(1.0e-20))
    blocked.clamp_(_IMPORTANCE_MIN, _IMPORTANCE_MAX).pow_(_IMPORTANCE_POWER)
    return blocked.reshape(-1, 8, 2, 4)


def _calibration_statistics(
    calibration: list, channels: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, int]:
    absmax = torch.zeros(channels, dtype=torch.float32, device=device)
    squares = torch.zeros_like(absmax)
    count = 0
    for item in calibration:
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            continue
        values = _restore_dense(item[0], item[1]).reshape(-1, channels)
        if _SMOOTH_STAT == "absmax":
            absmax = torch.maximum(absmax, values.abs().amax(dim=0))
        else:
            squares.add_(values.square().sum(dim=0))
        count += values.shape[0]
    if count == 0:
        absmax.fill_(1.0)
        squares.fill_(1.0)
        count = 1
    return absmax, squares, count


def _smooth_scale(
    weight: torch.Tensor, activation_absmax: torch.Tensor, activation_squares: torch.Tensor,
    activation_count: int,
) -> torch.Tensor:
    if _SMOOTH_STAT == "rms":
        activation_stat = (activation_squares / float(activation_count)).sqrt()
        weight_stat = weight.square().reshape(-1, weight.shape[-1]).mean(dim=0).sqrt()
    else:
        activation_stat = activation_absmax
        weight_stat = weight.abs().reshape(-1, weight.shape[-1]).amax(dim=0)
    scale = activation_stat.clamp_min(1.0e-8).pow(_SMOOTH_ALPHA)
    scale.div_(weight_stat.clamp_min(1.0e-8).pow(1.0 - _SMOOTH_ALPHA))
    if _SMOOTH_BLOCK_SHRINK != 1.0:
        blocked = scale.reshape(-1, HIF4_BLOCK_SIZE)
        center = blocked.log().mean(dim=-1, keepdim=True).exp()
        blocked.div_(center).pow_(_SMOOTH_BLOCK_SHRINK).mul_(center)
    return scale.clamp_(1.0 / _SMOOTH_LIMIT, _SMOOTH_LIMIT)


def _dequantize_hif4_params(params: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return (
        params["sign"] * params["mant"] * params["scale_lv3"]
        * params["scale_lv2"] * params["scale_factor"]
    ).flatten(-4)


def _hessian_local_scale_options(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    return lv2, lv3


def _build_sparse_block_hessians(
    transformed_samples: list[torch.Tensor],
    channels: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    rows = [
        sample.reshape(-1, channels).to(torch.float32)
        for sample in transformed_samples
        if isinstance(sample, torch.Tensor) and sample.numel()
    ]
    if not rows:
        return None
    matrix = torch.cat(rows, dim=0)
    if matrix.shape[0] < 2 or channels % HIF4_BLOCK_SIZE:
        return None
    block_count = channels // HIF4_BLOCK_SIZE

    def build(part: torch.Tensor) -> torch.Tensor:
        blocked = part.reshape(-1, block_count, HIF4_BLOCK_SIZE)
        hessian = torch.einsum("nbi,nbj->bij", blocked, blocked)
        hessian.div_(float(max(int(blocked.shape[0]), 1)))
        trace = torch.diagonal(hessian, dim1=-2, dim2=-1).sum(-1)
        scale = (trace / float(HIF4_BLOCK_SIZE)).clamp_min_(1.0e-8)
        eye = torch.eye(HIF4_BLOCK_SIZE, dtype=torch.float32, device=part.device)
        return hessian / scale[:, None, None] + _SPARSE_HESSIAN_DAMPING * eye

    midpoint = max(int(matrix.shape[0]) // 2, 1)
    left = matrix[:midpoint]
    right = matrix[midpoint:] if midpoint < int(matrix.shape[0]) else matrix
    return build(matrix), build(left), build(right)


def _hessian_loss_and_product(
    error: torch.Tensor, hessian: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    product = torch.bmm(error[:, None, :], hessian).squeeze(1)
    return (product * error).sum(dim=-1), product


def _initial_hessian_choices(
    flat_params: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    lv2 = flat_params["scale_lv2"].squeeze(-1).squeeze(-1)
    lv3 = flat_params["scale_lv3"].squeeze(-1)
    return (
        (lv2 > 1.5).to(torch.long) * 4
        + (lv3[:, :, 0] > 1.5).to(torch.long) * 2
        + (lv3[:, :, 1] > 1.5).to(torch.long)
    )


def _materialize_fixed_base_blocks(
    values: torch.Tensor,
    base_scale: torch.Tensor,
    choices: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    device = values.device
    lv2_options, lv3_options = _hessian_local_scale_options(device)
    blocked = values.reshape(-1, 8, 2, 4)
    lv2 = lv2_options[choices]
    lv3 = lv3_options[choices]
    total_scale = (
        base_scale[:, None, None, None]
        * lv2[:, :, None, None]
        * lv3[:, :, :, None]
    )
    mant = torch.round(blocked.abs() * (4.0 / total_scale)).clamp_(0.0, 7.0)
    mant.mul_(0.25)
    sign = torch.where(mant == 0, torch.zeros_like(blocked), torch.sign(blocked))
    reconstructed = (sign * mant * total_scale).reshape(-1, HIF4_BLOCK_SIZE)
    return {
        "scale_factor": base_scale[:, None, None, None],
        "scale_lv2": lv2[:, :, None, None],
        "scale_lv3": lv3[:, :, :, None],
        "sign": sign,
        "mant": mant,
    }, reconstructed


def _sweep_sparse_hessian_blocks(
    values: torch.Tensor,
    base_scale: torch.Tensor,
    choices: torch.Tensor,
    hessian: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    batch = int(values.shape[0])
    params, reconstructed = _materialize_fixed_base_blocks(
        values, base_scale, choices
    )
    del params
    error = values - reconstructed
    _, hessian_error = _hessian_loss_and_product(error, hessian)
    lv2_options, lv3_options = _hessian_local_scale_options(values.device)
    grouped_values = values.reshape(batch, 8, 2, 4)
    grouped_reconstructed = reconstructed.reshape(batch, 8, 8)
    grouped_error = error.reshape(batch, 8, 8)
    row_index = torch.arange(batch, device=values.device)

    for group in range(8):
        group_values = grouped_values[:, group]
        group_scale = (
            base_scale[:, None, None, None]
            * lv2_options[None, :, None, None]
            * lv3_options[None, :, :, None]
        )
        group_mant = torch.round(
            group_values.abs()[:, None, :, :] * (4.0 / group_scale)
        ).clamp_(0.0, 7.0)
        group_mant.mul_(0.25)
        group_sign = torch.where(
            group_mant == 0,
            torch.zeros_like(group_mant),
            torch.sign(group_values)[:, None, :, :],
        )
        candidates = (group_sign * group_mant * group_scale).reshape(batch, 8, 8)
        delta = candidates - grouped_reconstructed[:, group, None, :]
        start = group * 8
        stop = start + 8
        local_hessian = hessian[:, start:stop, start:stop]
        local_product = hessian_error[:, start:stop]
        delta_loss = -2.0 * (delta * local_product[:, None, :]).sum(dim=-1)
        delta_loss += torch.einsum("boi,bij,boj->bo", delta, local_hessian, delta)
        best_option = delta_loss.argmin(dim=1)
        best_delta = delta[row_index, best_option]
        grouped_reconstructed[:, group, :].add_(best_delta)
        grouped_error[:, group, :].sub_(best_delta)
        choices[:, group] = best_option
        hessian_columns = hessian[:, :, start:stop]
        hessian_error.sub_(
            torch.bmm(hessian_columns, best_delta[:, :, None]).squeeze(-1)
        )

    return _materialize_fixed_base_blocks(values, base_scale, choices)


def _apply_sparse_hessian_weight_refinement(
    values: torch.Tensor,
    weight_params: HiF4Params,
    transformed_samples: list[torch.Tensor],
) -> HiF4Params:
    rows, channels = int(values.shape[0]), int(values.shape[-1])
    block_count = channels // HIF4_BLOCK_SIZE
    hessians = _build_sparse_block_hessians(transformed_samples, channels)
    if hessians is None or block_count == 0:
        return weight_params
    full_hessian, left_hessian, right_hessian = hessians
    total_blocks = rows * block_count
    refined_count = int(total_blocks * _SPARSE_HESSIAN_TOP_FRACTION)
    if refined_count <= 0:
        return weight_params

    reconstructed = _dequantize_hif4_params(weight_params).reshape(rows, channels)
    values_blocks = values.reshape(rows, block_count, HIF4_BLOCK_SIZE)
    error_blocks = (values - reconstructed).reshape(
        rows, block_count, HIF4_BLOCK_SIZE
    )
    diagonal = torch.diagonal(full_hessian, dim1=-2, dim2=-1)
    pressure = (error_blocks.square() * diagonal.unsqueeze(0)).sum(dim=-1)
    pressure = torch.nan_to_num(pressure, nan=0.0, posinf=0.0, neginf=0.0)
    refined_count = min(max(refined_count, 1), total_blocks)
    selected = torch.topk(pressure.reshape(-1), refined_count).indices

    flat_params = {
        key: tensor.reshape(total_blocks, *tensor.shape[2:])
        for key, tensor in weight_params.items()
    }
    flat_reconstructed = reconstructed.reshape(rows, block_count, HIF4_BLOCK_SIZE)

    for start in range(0, refined_count, _SPARSE_HESSIAN_CHUNK_BLOCKS):
        end = min(start + _SPARSE_HESSIAN_CHUNK_BLOCKS, refined_count)
        flat_index = selected[start:end]
        row_index = flat_index // block_count
        block_index = flat_index % block_count
        batch_values = values_blocks[row_index, block_index]
        batch_base = flat_params["scale_factor"][flat_index].reshape(-1)
        batch_params = {key: value[flat_index] for key, value in flat_params.items()}
        choices = _initial_hessian_choices(batch_params)
        candidate_params, candidate_values = _sweep_sparse_hessian_blocks(
            batch_values,
            batch_base,
            choices,
            full_hessian[block_index],
        )
        baseline_error = batch_values - flat_reconstructed[row_index, block_index]
        candidate_error = batch_values - candidate_values
        baseline_full, _ = _hessian_loss_and_product(
            baseline_error, full_hessian[block_index]
        )
        candidate_full, _ = _hessian_loss_and_product(
            candidate_error, full_hessian[block_index]
        )
        baseline_left, _ = _hessian_loss_and_product(
            baseline_error, left_hessian[block_index]
        )
        candidate_left, _ = _hessian_loss_and_product(
            candidate_error, left_hessian[block_index]
        )
        baseline_right, _ = _hessian_loss_and_product(
            baseline_error, right_hessian[block_index]
        )
        candidate_right, _ = _hessian_loss_and_product(
            candidate_error, right_hessian[block_index]
        )
        accept = (
            (candidate_full < baseline_full * (1.0 - _SPARSE_HESSIAN_MIN_IMPROVEMENT))
            & (candidate_left < baseline_left * (1.0 - _SPARSE_HESSIAN_MIN_IMPROVEMENT))
            & (candidate_right < baseline_right * (1.0 - _SPARSE_HESSIAN_MIN_IMPROVEMENT))
        )
        if bool(accept.any()):
            accepted_index = flat_index[accept]
            for key, value in flat_params.items():
                value[accepted_index] = candidate_params[key][accept]

    return weight_params


@torch.inference_mode()
def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    weight = _restore_dense(weight_quant, weight_scale)
    channels = int(weight.shape[-1])
    rows = int(weight.shape[0]) if weight.ndim >= 2 else 0
    act_absmax, act_squares, act_count = _calibration_statistics(
        calib_activation_list, channels, weight.device
    )
    smooth = _smooth_scale(weight, act_absmax, act_squares, act_count)
    transformed_weight = _transform(weight, smooth, inverse=False)
    use_linear_plus = (
        rows >= _SPARSE_HESSIAN_MIN_WEIGHT_ROWS
        and channels >= _SPARSE_HESSIAN_MIN_CHANNELS
        and len(calib_activation_list) >= _SPARSE_HESSIAN_MIN_CALIBRATION_SAMPLES
    )
    use_sparse_hessian = (
        _USE_SPARSE_HESSIAN_WEIGHT_REFINEMENT
        and use_linear_plus
        and rows >= _SPARSE_HESSIAN_MIN_WEIGHT_ROWS
        and channels >= _SPARSE_HESSIAN_MIN_CHANNELS
    )
    transformed_activation_rows: list[torch.Tensor] = []
    if use_sparse_hessian:
        for item in calib_activation_list:
            if not isinstance(item, (tuple, list)) or len(item) < 2:
                continue
            activation = _restore_dense(item[0], item[1]).reshape(-1, channels)
            activation = _transform(activation, smooth, inverse=True).reshape(
                -1, channels
            )
            sample_rows = int(activation.shape[0])
            keep_rows = min(sample_rows, _SPARSE_HESSIAN_CALIBRATION_ROWS)
            if keep_rows <= 0:
                continue
            if keep_rows < sample_rows:
                index = torch.linspace(
                    0,
                    sample_rows - 1,
                    steps=keep_rows,
                    device=activation.device,
                ).round().to(torch.long)
                activation = activation.index_select(0, index)
            transformed_activation_rows.append(activation.contiguous())

    weight_importance = None
    if _USE_WEIGHT_IMPORTANCE and use_linear_plus:
        # The diagonal proxy for W' error uses E[A'^2] per transformed channel.
        transformed_squares = torch.zeros(
            channels, dtype=torch.float32, device=weight.device
        )
        transformed_count = 0
        for item in calib_activation_list:
            if not isinstance(item, (tuple, list)) or len(item) < 2:
                continue
            activation = _restore_dense(item[0], item[1])
            activation = _transform(activation, smooth, inverse=True).reshape(
                -1, channels
            )
            transformed_squares.add_(activation.square().sum(dim=0))
            transformed_count += activation.shape[0]
        if transformed_count:
            transformed_squares.div_(float(transformed_count))
        else:
            transformed_squares.fill_(1.0)
        weight_importance = _importance_from_channel_values(transformed_squares)
    weight_params = _quantize_dense(
        transformed_weight,
        weight_importance,
        _WEIGHT_BASE_OFFSETS if use_linear_plus else (0,),
    )
    if use_sparse_hessian:
        weight_params = _apply_sparse_hessian_weight_refinement(
            transformed_weight,
            weight_params,
            transformed_activation_rows,
        )
    activation_importance = None
    if _USE_ACTIVATION_IMPORTANCE and use_linear_plus:
        converted_weight = _dequantize_hif4_params(weight_params).float()
        column_energy = converted_weight.square().reshape(-1, channels).sum(dim=0)
        activation_importance = _importance_from_channel_values(column_energy)
    activation_state: dict[str, Any] = {
        "version": 4,
        "smooth": smooth,
        "use_activation_plus": use_linear_plus,
    }
    if activation_importance is not None:
        activation_state["column_energy"] = activation_importance
    return {"weight_params": weight_params, "activation_state": activation_state}


@torch.inference_mode()
def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_state: Any,
) -> HiF4Params:
    state = activation_state if isinstance(activation_state, Mapping) else {}
    values = _restore_dense(activation_quant, activation_scale)
    smooth = state.get("smooth")
    if not isinstance(smooth, torch.Tensor):
        smooth = torch.ones(values.shape[-1], dtype=torch.float32, device=values.device)
    else:
        smooth = smooth.to(device=values.device, dtype=torch.float32)
    transformed = _transform(values, smooth, inverse=True)
    use_activation_plus = bool(state.get("use_activation_plus", False))
    importance = (
        state.get("column_energy")
        if _USE_ACTIVATION_IMPORTANCE and use_activation_plus
        else None
    )
    base_offsets = _ACTIVATION_BASE_OFFSETS if use_activation_plus else (0,)
    return _quantize_dense(transformed, importance, base_offsets)


def _unpack_qk(item: Any) -> tuple[Any, Any]:
    if isinstance(item, Mapping):
        return item["q"], item["k"]
    if not isinstance(item, (tuple, list)) or len(item) < 4:
        raise ValueError("Attention calibration item must contain Q/K/V pairs")
    return item[0:2], item[2:4]


def _attention_transform(
    values: torch.Tensor,
    smooth: torch.Tensor,
    heads: int,
    head_dim: int,
    inverse: bool,
) -> torch.Tensor:
    if values.shape[-1] != heads * head_dim or head_dim % HIF4_BLOCK_SIZE:
        raise ValueError("Attention tensor has an incompatible flattened head layout")
    headed = values.reshape(-1, heads, head_dim)
    local = smooth.to(device=values.device, dtype=torch.float32)
    headed = headed / local if inverse else headed * local
    return _hadamard64(headed.reshape_as(values))


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
    repeats = q_num_heads // kv_num_heads
    q_moments: list[torch.Tensor] = []
    k_moments: list[torch.Tensor] = []
    sample_logs: list[torch.Tensor] = []
    for item in calib_qkv_list:
        q_pair, k_pair = _unpack_qk(item)
        q = _restore_dense(*q_pair).reshape(-1, q_num_heads, head_dim)
        k = _restore_dense(*k_pair).reshape(-1, kv_num_heads, head_dim)
        q_moment = q.square().mean(dim=0)
        k_moment = k.square().mean(dim=0)
        grouped_q = q_moment.reshape(
            kv_num_heads, repeats, head_dim
        ).mean(dim=1)
        sample_smooth = (
            grouped_q.clamp_min(1.0e-20) / k_moment.clamp_min(1.0e-20)
        ).pow(0.125)
        sample_logs.append(torch.log2(sample_smooth.clamp(1.0 / 16.0, 16.0)))
        q_moments.append(q_moment)
        k_moments.append(k_moment)

    apply_transform = False
    log_smooth_std = 0.0
    cross_sample_std = float("inf")
    smooth_q: torch.Tensor | None = None
    smooth_k: torch.Tensor | None = None
    if q_moments:
        q_moment = torch.stack(q_moments).mean(dim=0)
        k_moment = torch.stack(k_moments).mean(dim=0)
        grouped_q = q_moment.reshape(
            kv_num_heads, repeats, head_dim
        ).mean(dim=1)
        smooth_k = (
            grouped_q.clamp_min(1.0e-20) / k_moment.clamp_min(1.0e-20)
        ).pow(0.125).clamp_(1.0 / 16.0, 16.0)
        smooth_q = smooth_k.repeat_interleave(repeats, dim=0)
        aggregate_log = torch.log2(smooth_k)
        log_smooth_std = float(aggregate_log.std().item())
        if len(sample_logs) >= 2:
            cross_sample_std = float(
                torch.stack(sample_logs).std(dim=0).mean().item()
            )
        apply_transform = (
            repeats >= _ATTENTION_MIN_GQA_RATIO
            and len(sample_logs) >= _ATTENTION_MIN_CALIBRATION_SAMPLES
            and log_smooth_std <= _ATTENTION_MAX_LOG_SMOOTH_STD
            and cross_sample_std <= _ATTENTION_MAX_CROSS_SAMPLE_STD
        )

    q_state = {
        "version": 5,
        "apply_transform": apply_transform,
        "q_base_search": bool(
            apply_transform
            and q_num_heads == 16
            and kv_num_heads == 2
            and head_dim == 256
            and len(calib_qkv_list) >= 5
        ),
        "smooth": smooth_q if apply_transform else None,
        "log_smooth_std": log_smooth_std,
        "cross_sample_std": cross_sample_std if cross_sample_std != float("inf") else None,
    }
    k_state = {
        "version": 5,
        "apply_transform": apply_transform,
        "smooth": smooth_k if apply_transform else None,
        "log_smooth_std": log_smooth_std,
        "cross_sample_std": cross_sample_std if cross_sample_std != float("inf") else None,
    }
    return {
        "q_state": q_state,
        "k_state": k_state,
        "v_state": {
            "version": 5,
            "base_search": 3 if (_USE_MULTIBASE_V and apply_transform) else 1,
            "v_base_search": bool(
                apply_transform
                and q_num_heads == 16
                and kv_num_heads == 2
                and head_dim == 256
                and len(calib_qkv_list) >= 5
            ),
        },
    }


@torch.inference_mode()
def hif4_dynamic_quantize_q(
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    q_num_heads: int,
    head_dim: int,
    q_state: Any,
) -> HiF4Params:
    values = _restore_dense(q_quant, q_scale)
    state = q_state if isinstance(q_state, Mapping) else {}
    smooth = state.get("smooth")
    if state.get("apply_transform") and isinstance(smooth, torch.Tensor):
        values = _attention_transform(
            values, smooth, q_num_heads, head_dim, inverse=True
        )
    if state.get("q_base_search"):
        return _quantize_dense(
            values,
            base_offsets=_ATTENTION_Q_BASE_OFFSETS,
            min_relative_improvement=_ATTENTION_Q_MIN_IMPROVEMENT,
        )
    return _quantize_dense(values)


@torch.inference_mode()
def hif4_dynamic_quantize_k(
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    k_state: Any,
) -> HiF4Params:
    values = _restore_dense(k_quant, k_scale)
    state = k_state if isinstance(k_state, Mapping) else {}
    smooth = state.get("smooth")
    if state.get("apply_transform") and isinstance(smooth, torch.Tensor):
        values = _attention_transform(
            values, smooth, kv_num_heads, head_dim, inverse=False
        )
    return _quantize_dense(values)


@torch.inference_mode()
def hif4_dynamic_quantize_v(
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    v_state: Any,
) -> HiF4Params:
    del kv_num_heads, head_dim
    values = _restore_dense(v_quant, v_scale)
    state = v_state if isinstance(v_state, Mapping) else {}
    if _USE_MULTIBASE_V and state.get("v_base_search"):
        return _quantize_dense(
            values,
            base_offsets=_ATTENTION_V_BASE_OFFSETS,
            min_relative_improvement=_ATTENTION_V_MIN_IMPROVEMENT,
        )
    if _USE_MULTIBASE_V and state.get("base_search") == 3:
        return _quantize_dense(values, base_offsets=(0, -1, 1), min_relative_improvement=0.01)
    return _quantize_dense(values)
