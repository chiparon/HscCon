"""HiF4 competition submission interface.

The six public functions in this module form the submission boundary.  Inputs are
NVFP4 value carriers plus per-16-element block scales.  Quantization functions
must return the five tensors described by :class:`HiF4Params`.

The quantization policy is intentionally not implemented yet: later competition
material will be incorporated without changing these public signatures.
"""

from __future__ import annotations

from typing import Any, TypedDict

import torch


NVFP4_BLOCK_SIZE = 16
HIF4_BLOCK_SIZE = 64


class HiF4Params(TypedDict):
    """Logical HiF4 decomposition for an input shaped ``(*prefix, C)``.

    The output shapes are ``(*prefix, C // 64, 1, 1, 1)`` for ``scale_factor``,
    ``(*prefix, C // 64, 8, 1, 1)`` for ``scale_lv2``,
    ``(*prefix, C // 64, 8, 2, 1)`` for ``scale_lv3``, and
    ``(*prefix, C // 64, 8, 2, 4)`` for both ``sign`` and ``mant``.
    """

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
    """Restore an NVFP4 carrier and its block scales to a BF16 tensor."""
    if quant_float.ndim == 0 or scale_float.ndim == 0:
        raise ValueError("NVFP4 carrier and scale must have at least one dimension")
    if blk_size <= 0:
        raise ValueError(f"block size must be positive, got {blk_size}")

    channels = int(quant_float.shape[-1])
    if channels % blk_size != 0:
        raise ValueError(
            f"last dimension {channels} is not divisible by block size {blk_size}"
        )

    expected_scale_shape = quant_float.shape[:-1] + (channels // blk_size,)
    if scale_float.shape != expected_scale_shape:
        raise ValueError(
            f"scale shape {tuple(scale_float.shape)} does not match expected "
            f"shape {tuple(expected_scale_shape)}"
        )

    blocked = quant_float.unflatten(-1, (-1, blk_size))
    restored = blocked * scale_float.unsqueeze(-1)
    return restored.flatten(-2, -1).to(torch.bfloat16)


def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    """Calibrate a Linear layer and return its weight parameters and state.

    ``weight_params`` must be a :class:`HiF4Params`; ``activation_state`` must
    contain only portable data needed by dynamic activation quantization.
    """
    raise NotImplementedError(
        "Implement hif4_calibration_and_quantize_weight in your solution.py"
    )


def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_state: Any,
) -> HiF4Params:
    """Dynamically quantize a Linear activation using calibration state.

    Compliance boundary: an implementation must never compute (or approximate
    through an equivalent operator) ``activation @ weight`` and use that result
    to fit, search for, or infer the activation quantization.  The online path
    should remain bounded and vectorizable because inference latency is a scored
    objective in later competition stages.
    """
    raise NotImplementedError(
        "Implement hif4_dynamic_quantize_activation in your solution.py"
    )


def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    """Return portable ``q_state``, ``k_state`` and ``v_state`` values."""
    raise NotImplementedError("Implement hif4_calibration_attention in your solution.py")


def hif4_dynamic_quantize_q(
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    q_num_heads: int,
    head_dim: int,
    q_state: Any,
) -> HiF4Params:
    """Dynamically quantize a Query tensor to HiF4 parameters."""
    raise NotImplementedError("Implement hif4_dynamic_quantize_q in your solution.py")


def hif4_dynamic_quantize_k(
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    k_state: Any,
) -> HiF4Params:
    """Dynamically quantize a Key tensor to HiF4 parameters."""
    raise NotImplementedError("Implement hif4_dynamic_quantize_k in your solution.py")


def hif4_dynamic_quantize_v(
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    v_state: Any,
) -> HiF4Params:
    """Dynamically quantize a Value tensor to HiF4 parameters."""
    raise NotImplementedError("Implement hif4_dynamic_quantize_v in your solution.py")
