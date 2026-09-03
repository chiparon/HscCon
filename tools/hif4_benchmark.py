"""Reproducible correctness and latency benchmark for HiF4 submissions.

The benchmark deliberately builds all synthetic NVFP4 tensors before starting
any timer.  Two submissions are then run in a deterministic, shuffled,
interleaved order so that first-run and thermal effects are not assigned to the
same submission on every invocation.

The calibration container contract used by this repository is:

* Linear calibration item: ``(activation_quant, activation_scale)``.
* Attention calibration item: ``(q_quant, q_scale, k_quant, k_scale,
  v_quant, v_scale)``.

Run ``python tools/hif4_benchmark.py --help`` for the command-line interface.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch


NVFP4_LEVELS = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0))
HIF4_MANTISSA_LEVELS = torch.arange(8, dtype=torch.float32) * 0.25
NVFP4_BLOCK_SIZE = 16
HIF4_BLOCK_SIZE = 64
E6M2_EXPONENT_MIN = -48
E6M2_EXPONENT_MAX = 15

PUBLIC_FUNCTIONS = (
    "hif4_calibration_and_quantize_weight",
    "hif4_dynamic_quantize_activation",
    "hif4_calibration_attention",
    "hif4_dynamic_quantize_q",
    "hif4_dynamic_quantize_k",
    "hif4_dynamic_quantize_v",
)


@dataclass(frozen=True)
class NVFP4Tensor:
    """An NVFP4 value carrier and its per-16-value decoded E4M3 scale."""

    quant: torch.Tensor
    scale: torch.Tensor

    @property
    def shape(self) -> torch.Size:
        return self.quant.shape


@dataclass(frozen=True)
class LinearCase:
    name: str
    weight: NVFP4Tensor
    calibration: tuple[NVFP4Tensor, ...]
    test: tuple[NVFP4Tensor, ...]


@dataclass(frozen=True)
class AttentionCase:
    name: str
    q_num_heads: int
    kv_num_heads: int
    head_dim: int
    calibration: tuple[tuple[NVFP4Tensor, NVFP4Tensor, NVFP4Tensor], ...]
    test: tuple[tuple[NVFP4Tensor, NVFP4Tensor, NVFP4Tensor], ...]


@dataclass(frozen=True)
class SyntheticSuite:
    seed: int
    profile: str
    linear: tuple[LinearCase, ...]
    attention: tuple[AttentionCase, ...]


@dataclass(frozen=True)
class LatencySummary:
    calls: int
    median_ms: float
    p95_ms: float
    total_ms: float


@dataclass(frozen=True)
class TimingReport:
    overall: LatencySummary
    median_suite_ms: float
    p95_suite_ms: float
    by_function: dict[str, LatencySummary]


@dataclass(frozen=True)
class QualityReport:
    linear_mse: dict[str, float]
    attention_mse: dict[str, float]

    @property
    def mean_linear_mse(self) -> float:
        return statistics.fmean(self.linear_mse.values()) if self.linear_mse else 0.0

    @property
    def mean_attention_mse(self) -> float:
        return (
            statistics.fmean(self.attention_mse.values())
            if self.attention_mse
            else 0.0
        )


@dataclass(frozen=True)
class ComparisonReport:
    baseline_path: str
    candidate_path: str
    seed: int
    profile: str
    repeats: int
    warmup: int
    data_generation_ms: float
    baseline_timing: TimingReport
    candidate_timing: TimingReport
    baseline_quality: QualityReport
    candidate_quality: QualityReport
    relative_speedup: float

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["baseline_quality"]["mean_linear_mse"] = (
            self.baseline_quality.mean_linear_mse
        )
        result["baseline_quality"]["mean_attention_mse"] = (
            self.baseline_quality.mean_attention_mse
        )
        result["candidate_quality"]["mean_linear_mse"] = (
            self.candidate_quality.mean_linear_mse
        )
        result["candidate_quality"]["mean_attention_mse"] = (
            self.candidate_quality.mean_attention_mse
        )
        return result


@dataclass
class _PreparedSolution:
    module: ModuleType | Any
    linear_results: dict[str, Mapping[str, Any]]
    attention_states: dict[str, Mapping[str, Any]]


@dataclass(frozen=True)
class _Invocation:
    key: tuple[str, str, int]
    function_name: str
    call: Callable[[], Any]
    device: torch.device


class HiF4ValidationError(ValueError):
    """Raised when a submission returns malformed or illegal HiF4 tensors."""


def _e4m3_round_scale(scale: torch.Tensor) -> torch.Tensor:
    """Round positive scales to finite E4M3 values without requiring FP8 ops."""
    # NVFP4 uses decoded E4M3 block scales.  The synthetic magnitudes stay in
    # the normal range, so this normal-only implementation is sufficient and
    # works on PyTorch builds without float8 CPU kernels.
    scale = scale.clamp(min=2.0**-6, max=448.0)
    exponent = torch.floor(torch.log2(scale))
    significand = scale / torch.exp2(exponent)
    significand = torch.round(significand * 8.0) / 8.0
    carry = significand >= 2.0
    exponent = exponent + carry
    significand = torch.where(carry, torch.ones_like(significand), significand)
    rounded = torch.exp2(exponent) * significand
    return rounded.clamp(max=448.0)


def quantize_synthetic_nvfp4(values: torch.Tensor) -> NVFP4Tensor:
    """Quantize a CPU floating tensor to an NVFP4 carrier and decoded scales."""
    if values.ndim == 0 or values.shape[-1] % NVFP4_BLOCK_SIZE:
        raise ValueError("NVFP4 input last dimension must be divisible by 16")
    values = values.to(dtype=torch.float32, device="cpu")
    blocks = values.unflatten(-1, (-1, NVFP4_BLOCK_SIZE))
    amax = blocks.abs().amax(dim=-1)
    raw_scale = torch.where(amax > 0, amax / 6.0, torch.ones_like(amax))
    scale = _e4m3_round_scale(raw_scale)
    normalized = (blocks / scale.unsqueeze(-1)).clamp(-6.0, 6.0)
    abs_normalized = normalized.abs().unsqueeze(-1)
    level_indices = (abs_normalized - NVFP4_LEVELS).abs().argmin(dim=-1)
    carrier = NVFP4_LEVELS[level_indices] * torch.sign(normalized)
    carrier = carrier.flatten(-2, -1).contiguous()
    return NVFP4Tensor(carrier, scale.contiguous())


def dequantize_nvfp4(value: NVFP4Tensor) -> torch.Tensor:
    """Dequantize an :class:`NVFP4Tensor` to float32."""
    quant, scale = value.quant, value.scale
    if quant.ndim == 0 or quant.shape[-1] % NVFP4_BLOCK_SIZE:
        raise ValueError("NVFP4 carrier last dimension must be divisible by 16")
    expected = quant.shape[:-1] + (quant.shape[-1] // NVFP4_BLOCK_SIZE,)
    if scale.shape != expected:
        raise ValueError(
            f"NVFP4 scale shape {tuple(scale.shape)} != expected {tuple(expected)}"
        )
    return (
        quant.to(torch.float32)
        .unflatten(-1, (-1, NVFP4_BLOCK_SIZE))
        .mul(scale.to(torch.float32).unsqueeze(-1))
        .flatten(-2, -1)
    )


def _normal_samples(
    shape: Sequence[int], generator: torch.Generator, outlier_rate: float
) -> torch.Tensor:
    values = torch.randn(tuple(shape), generator=generator, dtype=torch.float32)
    channels = shape[-1]
    channel_gain = torch.exp(
        torch.linspace(-1.15, 1.15, channels, dtype=torch.float32)
    )
    values.mul_(channel_gain)
    if outlier_rate > 0:
        mask = torch.rand(tuple(shape), generator=generator) < outlier_rate
        signs = torch.where(
            torch.rand(tuple(shape), generator=generator) < 0.5, -1.0, 1.0
        )
        magnitudes = 10.0 + 10.0 * torch.rand(tuple(shape), generator=generator)
        values = torch.where(mask, signs * magnitudes * channel_gain, values)
    return values


def _make_nvfp4(
    shape: Sequence[int], generator: torch.Generator, outlier_rate: float, device: str
) -> NVFP4Tensor:
    item = quantize_synthetic_nvfp4(_normal_samples(shape, generator, outlier_rate))
    return NVFP4Tensor(item.quant.to(device), item.scale.to(device))


def build_synthetic_suite(
    seed: int = 20260903, profile: str = "default", device: str = "cpu"
) -> SyntheticSuite:
    """Build deterministic normal/outlier Linear and MHA/GQA Attention cases.

    Tensor generation and host-to-device transfer finish before this function
    returns, and therefore cannot enter any algorithm timing window.
    """
    if profile not in {"smoke", "default", "stress"}:
        raise ValueError("profile must be one of: smoke, default, stress")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    if profile == "smoke":
        linear_specs = (
            ("linear_normal_64", 16, 32, 64, 0.0),
            ("linear_outlier_128", 24, 32, 128, 0.015),
        )
        attention_specs = (
            ("attention_mha", 1, 4, 4, 64, 4, 4, 0.0),
            ("attention_gqa_outlier", 1, 8, 2, 64, 4, 6, 0.01),
        )
        calibration_count, test_count = 1, 1
    elif profile == "default":
        linear_specs = (
            ("linear_normal_128", 96, 24, 128, 0.0),
            ("linear_outlier_256", 160, 32, 256, 0.008),
            ("linear_wide_512", 128, 48, 512, 0.002),
        )
        attention_specs = (
            ("attention_mha_short", 1, 4, 4, 64, 12, 12, 0.0),
            ("attention_gqa", 2, 8, 2, 64, 10, 14, 0.002),
            ("attention_gqa_outlier", 1, 8, 2, 128, 12, 20, 0.008),
        )
        calibration_count, test_count = 2, 2
    else:
        linear_specs = (
            ("linear_normal_256", 256, 96, 256, 0.0),
            ("linear_outlier_512", 512, 128, 512, 0.006),
            ("linear_wide_1024", 512, 128, 1024, 0.002),
            ("linear_tall_2048", 2048, 64, 256, 0.004),
        )
        attention_specs = (
            ("attention_mha", 2, 8, 8, 64, 32, 32, 0.0),
            ("attention_gqa_4to1", 2, 16, 4, 64, 24, 64, 0.002),
            ("attention_gqa_8to1_outlier", 1, 16, 2, 128, 32, 96, 0.006),
        )
        calibration_count, test_count = 3, 3

    linear_cases: list[LinearCase] = []
    for name, output_features, tokens, channels, outlier_rate in linear_specs:
        weight = _make_nvfp4(
            (output_features, channels), generator, outlier_rate * 0.5, device
        )
        calibration = tuple(
            _make_nvfp4((tokens, channels), generator, outlier_rate, device)
            for _ in range(calibration_count)
        )
        test = tuple(
            _make_nvfp4((tokens + index * 3, channels), generator, outlier_rate, device)
            for index in range(test_count)
        )
        linear_cases.append(LinearCase(name, weight, calibration, test))

    attention_cases: list[AttentionCase] = []
    for (
        name,
        batch,
        q_heads,
        kv_heads,
        head_dim,
        q_tokens,
        kv_tokens,
        outlier_rate,
    ) in attention_specs:
        def make_qkv(extra_tokens: int = 0) -> tuple[NVFP4Tensor, ...]:
            q = _make_nvfp4(
                (batch, q_tokens + extra_tokens, q_heads * head_dim),
                generator,
                outlier_rate,
                device,
            )
            k = _make_nvfp4(
                (batch, kv_tokens + extra_tokens, kv_heads * head_dim),
                generator,
                outlier_rate,
                device,
            )
            v = _make_nvfp4(
                (batch, kv_tokens + extra_tokens, kv_heads * head_dim),
                generator,
                outlier_rate,
                device,
            )
            return q, k, v

        calibration = tuple(make_qkv() for _ in range(calibration_count))
        test = tuple(make_qkv(index * 2) for index in range(test_count))
        attention_cases.append(
            AttentionCase(
                name,
                q_heads,
                kv_heads,
                head_dim,
                calibration,  # type: ignore[arg-type]
                test,  # type: ignore[arg-type]
            )
        )

    if str(device).startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))
    return SyntheticSuite(
        seed, profile, tuple(linear_cases), tuple(attention_cases)
    )


def _require_tensor(params: Mapping[str, Any], key: str) -> torch.Tensor:
    value = params.get(key)
    if not isinstance(value, torch.Tensor):
        raise HiF4ValidationError(f"{key!r} must be a torch.Tensor")
    return value


def e6m2_legal_mask(values: torch.Tensor, allow_nan: bool = True) -> torch.Tensor:
    """Return a mask for unsigned HiF4 E6M2 decoded values.

    E6M2 has exponent bias 48, supports only normal exponents ``[-48, 15]``,
    and has two fraction bits.  Encoding ``111111_11`` is NaN, so exponent 15
    has only significands 1.0, 1.25 and 1.5.  E6M2 has no numeric zero.
    """
    values = values.to(torch.float32)
    is_nan = torch.isnan(values)
    positive_finite = torch.isfinite(values) & (values > 0)
    safe = torch.where(positive_finite, values, torch.ones_like(values))
    exponent = torch.floor(torch.log2(safe))
    significand_steps = safe / torch.exp2(exponent) * 4.0
    on_grid = significand_steps == torch.round(significand_steps)
    in_significand_range = (significand_steps >= 4.0) & (
        significand_steps <= 7.0
    )
    in_exponent_range = (exponent >= E6M2_EXPONENT_MIN) & (
        exponent <= E6M2_EXPONENT_MAX
    )
    not_reserved_nan = ~(
        (exponent == E6M2_EXPONENT_MAX) & (significand_steps == 7.0)
    )
    normal = (
        positive_finite
        & on_grid
        & in_significand_range
        & in_exponent_range
        & not_reserved_nan
    )
    return normal | (is_nan & allow_nan)


def validate_hif4_params(
    params: Mapping[str, Any], original_shape: Sequence[int], allow_nan_scale: bool = True
) -> None:
    """Validate field shapes and exact legal decoded values for one HiF4 result."""
    if not isinstance(params, Mapping):
        raise HiF4ValidationError("HiF4 params must be a mapping")
    if not original_shape or original_shape[-1] % HIF4_BLOCK_SIZE:
        raise ValueError("original last dimension must be divisible by 64")
    prefix = tuple(original_shape[:-1])
    blocks = original_shape[-1] // HIF4_BLOCK_SIZE
    expected_shapes = {
        "scale_factor": prefix + (blocks, 1, 1, 1),
        "scale_lv2": prefix + (blocks, 8, 1, 1),
        "scale_lv3": prefix + (blocks, 8, 2, 1),
        "sign": prefix + (blocks, 8, 2, 4),
        "mant": prefix + (blocks, 8, 2, 4),
    }
    tensors: dict[str, torch.Tensor] = {}
    for key, expected in expected_shapes.items():
        tensor = _require_tensor(params, key)
        if tuple(tensor.shape) != expected:
            raise HiF4ValidationError(
                f"{key} shape {tuple(tensor.shape)} != expected {expected}"
            )
        tensors[key] = tensor

    checks = {
        "scale_factor": e6m2_legal_mask(
            tensors["scale_factor"], allow_nan=allow_nan_scale
        ),
        "scale_lv2": (tensors["scale_lv2"] == 1)
        | (tensors["scale_lv2"] == 2),
        "scale_lv3": (tensors["scale_lv3"] == 1)
        | (tensors["scale_lv3"] == 2),
        "sign": (tensors["sign"] == -1)
        | (tensors["sign"] == 0)
        | (tensors["sign"] == 1),
        "mant": torch.isfinite(tensors["mant"])
        & (tensors["mant"] >= 0)
        & (tensors["mant"] <= 1.75)
        & (tensors["mant"] * 4 == torch.round(tensors["mant"] * 4)),
    }
    for key, mask in checks.items():
        if not bool(mask.all().item()):
            invalid_count = int((~mask).sum().item())
            raise HiF4ValidationError(
                f"{key} contains {invalid_count} values outside its legal set"
            )


def dequantize_hif4(
    params: Mapping[str, Any],
    original_shape: Sequence[int],
    validate: bool = True,
) -> torch.Tensor:
    """Validate and dequantize logical HiF4 fields to float32.

    The official all-zero-block sentinel is the E6M2 NaN encoding.  It is
    interpreted as a zero base scale here instead of propagating IEEE NaNs.
    """
    if validate:
        validate_hif4_params(params, original_shape)
    scale_factor = _require_tensor(params, "scale_factor").to(torch.float32)
    scale_factor = torch.nan_to_num(scale_factor, nan=0.0)
    restored = (
        _require_tensor(params, "sign").to(torch.float32)
        * _require_tensor(params, "mant").to(torch.float32)
        * _require_tensor(params, "scale_lv3").to(torch.float32)
        * _require_tensor(params, "scale_lv2").to(torch.float32)
        * scale_factor
    )
    return restored.reshape(tuple(original_shape))


def linear_operator_mse(
    activation: NVFP4Tensor,
    weight: NVFP4Tensor,
    activation_params: Mapping[str, Any],
    weight_params: Mapping[str, Any],
) -> float:
    """MSE of ``A @ W.T`` using source NVFP4 and converted HiF4 operands."""
    source_a = dequantize_nvfp4(activation)
    source_w = dequantize_nvfp4(weight)
    converted_a = dequantize_hif4(activation_params, activation.shape)
    converted_w = dequantize_hif4(weight_params, weight.shape)
    reference = source_a @ source_w.transpose(-1, -2)
    actual = converted_a @ converted_w.transpose(-1, -2)
    return float(torch.mean((reference - actual).square()).item())


def _attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    if q_heads % kv_heads:
        raise ValueError("q_num_heads must be divisible by kv_num_heads for GQA")
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("synthetic Attention tensors must have shape (B, T, H*D)")
    batch, q_tokens, _ = q.shape
    kv_tokens = k.shape[1]
    q = q.reshape(batch, q_tokens, q_heads, head_dim).transpose(1, 2)
    k = k.reshape(batch, kv_tokens, kv_heads, head_dim).transpose(1, 2)
    v = v.reshape(batch, kv_tokens, kv_heads, head_dim).transpose(1, 2)
    if q_heads != kv_heads:
        repeats = q_heads // kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
    logits = (q @ k.transpose(-1, -2)) * (head_dim**-0.5)
    probabilities = torch.softmax(logits, dim=-1)
    return (probabilities @ v).transpose(1, 2).reshape(
        batch, q_tokens, q_heads * head_dim
    )


def attention_operator_mse(
    qkv: tuple[NVFP4Tensor, NVFP4Tensor, NVFP4Tensor],
    qkv_params: tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]],
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> float:
    """MSE of scaled-dot-product MHA/GQA for source and converted Q/K/V."""
    q, k, v = qkv
    qp, kp, vp = qkv_params
    reference = _attention(
        dequantize_nvfp4(q),
        dequantize_nvfp4(k),
        dequantize_nvfp4(v),
        q_num_heads,
        kv_num_heads,
        head_dim,
    )
    actual = _attention(
        dequantize_hif4(qp, q.shape),
        dequantize_hif4(kp, k.shape),
        dequantize_hif4(vp, v.shape),
        q_num_heads,
        kv_num_heads,
        head_dim,
    )
    return float(torch.mean((reference - actual).square()).item())


def _linear_calibration_payload(case: LinearCase) -> list[tuple[torch.Tensor, ...]]:
    return [(item.quant, item.scale) for item in case.calibration]


def _attention_calibration_payload(
    case: AttentionCase,
) -> list[tuple[torch.Tensor, ...]]:
    return [
        (q.quant, q.scale, k.quant, k.scale, v.quant, v.scale)
        for q, k, v in case.calibration
    ]


def _load_solution(path: str | Path, role: str) -> ModuleType:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{role} solution does not exist: {resolved}")
    module_name = f"_hif4_bench_{role}_{abs(hash(str(resolved)))}_{time.time_ns()}"
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load Python module from {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    missing = [name for name in PUBLIC_FUNCTIONS if not callable(getattr(module, name, None))]
    if missing:
        raise AttributeError(f"{resolved} is missing functions: {', '.join(missing)}")
    return module


def _prepare_solution(module: ModuleType | Any, suite: SyntheticSuite) -> _PreparedSolution:
    linear_results: dict[str, Mapping[str, Any]] = {}
    for case in suite.linear:
        result = module.hif4_calibration_and_quantize_weight(
            case.weight.quant,
            case.weight.scale,
            _linear_calibration_payload(case),
        )
        if not isinstance(result, Mapping):
            raise TypeError(f"{case.name}: Linear calibration result must be a mapping")
        if "weight_params" not in result or "activation_state" not in result:
            raise KeyError(
                f"{case.name}: Linear calibration needs weight_params and activation_state"
            )
        linear_results[case.name] = result

    attention_states: dict[str, Mapping[str, Any]] = {}
    for case in suite.attention:
        result = module.hif4_calibration_attention(
            _attention_calibration_payload(case),
            case.q_num_heads,
            case.kv_num_heads,
            case.head_dim,
        )
        if not isinstance(result, Mapping):
            raise TypeError(f"{case.name}: Attention calibration result must be a mapping")
        missing = {"q_state", "k_state", "v_state"} - set(result)
        if missing:
            raise KeyError(f"{case.name}: Attention calibration missing {sorted(missing)}")
        attention_states[case.name] = result
    return _PreparedSolution(module, linear_results, attention_states)


def _make_invocations(
    prepared: _PreparedSolution, suite: SyntheticSuite
) -> dict[tuple[str, str, int], _Invocation]:
    module = prepared.module
    invocations: dict[tuple[str, str, int], _Invocation] = {}
    for case in suite.linear:
        calibration_payload = _linear_calibration_payload(case)
        key = ("hif4_calibration_and_quantize_weight", case.name, 0)
        invocations[key] = _Invocation(
            key,
            key[0],
            lambda case=case, payload=calibration_payload: module.hif4_calibration_and_quantize_weight(
                case.weight.quant, case.weight.scale, payload
            ),
            case.weight.quant.device,
        )
        state = prepared.linear_results[case.name]["activation_state"]
        for index, activation in enumerate(case.test):
            key = ("hif4_dynamic_quantize_activation", case.name, index)
            invocations[key] = _Invocation(
                key,
                key[0],
                lambda activation=activation, state=state: module.hif4_dynamic_quantize_activation(
                    activation.quant, activation.scale, state
                ),
                activation.quant.device,
            )

    for case in suite.attention:
        calibration_payload = _attention_calibration_payload(case)
        key = ("hif4_calibration_attention", case.name, 0)
        invocations[key] = _Invocation(
            key,
            key[0],
            lambda case=case, payload=calibration_payload: module.hif4_calibration_attention(
                payload, case.q_num_heads, case.kv_num_heads, case.head_dim
            ),
            case.calibration[0][0].quant.device,
        )
        states = prepared.attention_states[case.name]
        function_specs = (
            ("hif4_dynamic_quantize_q", 0, case.q_num_heads, states["q_state"]),
            ("hif4_dynamic_quantize_k", 1, case.kv_num_heads, states["k_state"]),
            ("hif4_dynamic_quantize_v", 2, case.kv_num_heads, states["v_state"]),
        )
        for function_name, component, heads, state in function_specs:
            function = getattr(module, function_name)
            for index, qkv in enumerate(case.test):
                item = qkv[component]
                key = (function_name, case.name, index)
                invocations[key] = _Invocation(
                    key,
                    function_name,
                    lambda item=item, heads=heads, state=state, function=function, case=case: function(
                        item.quant, item.scale, heads, case.head_dim, state
                    ),
                    item.quant.device,
                )
    return invocations


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _time_call(invocation: _Invocation) -> float:
    _synchronize(invocation.device)
    start = time.perf_counter_ns()
    invocation.call()
    _synchronize(invocation.device)
    return (time.perf_counter_ns() - start) / 1_000_000.0


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _summarize(values: Sequence[float]) -> LatencySummary:
    return LatencySummary(
        len(values),
        statistics.median(values) if values else 0.0,
        _percentile(values, 0.95),
        sum(values),
    )


def _timing_report(
    records: Sequence[tuple[int, str, float]], repeats: int
) -> TimingReport:
    all_values = [duration for _, _, duration in records]
    by_function_values: dict[str, list[float]] = defaultdict(list)
    by_repeat: dict[int, float] = defaultdict(float)
    for repeat, function_name, duration in records:
        by_function_values[function_name].append(duration)
        by_repeat[repeat] += duration
    suite_values = [by_repeat[index] for index in range(repeats)]
    return TimingReport(
        _summarize(all_values),
        statistics.median(suite_values) if suite_values else 0.0,
        _percentile(suite_values, 0.95),
        {name: _summarize(values) for name, values in sorted(by_function_values.items())},
    )


def _run_interleaved_timing(
    baseline: _PreparedSolution,
    candidate: _PreparedSolution,
    suite: SyntheticSuite,
    repeats: int,
    warmup: int,
    seed: int,
) -> tuple[TimingReport, TimingReport]:
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be >= 1 and warmup must be >= 0")
    invocation_sets = (
        _make_invocations(baseline, suite),
        _make_invocations(candidate, suite),
    )
    if invocation_sets[0].keys() != invocation_sets[1].keys():
        raise RuntimeError("baseline and candidate invocation sets differ")
    keys = list(invocation_sets[0])
    rng = random.Random(seed)

    # Warm-up is deliberately untimed.  Alternate which solution warms first.
    for warmup_index in range(warmup):
        shuffled = keys.copy()
        rng.shuffle(shuffled)
        for index, key in enumerate(shuffled):
            order = (0, 1) if (warmup_index + index) % 2 == 0 else (1, 0)
            for solution_index in order:
                invocation = invocation_sets[solution_index][key]
                invocation.call()
                _synchronize(invocation.device)

    records: tuple[list[tuple[int, str, float]], list[tuple[int, str, float]]] = (
        [],
        [],
    )
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for repeat in range(repeats):
            shuffled = keys.copy()
            rng.shuffle(shuffled)
            for key in shuffled:
                order = [0, 1]
                rng.shuffle(order)
                for solution_index in order:
                    invocation = invocation_sets[solution_index][key]
                    duration = _time_call(invocation)
                    records[solution_index].append(
                        (repeat, invocation.function_name, duration)
                    )
    finally:
        if gc_was_enabled:
            gc.enable()
    return _timing_report(records[0], repeats), _timing_report(records[1], repeats)


def evaluate_quality(
    prepared: _PreparedSolution, suite: SyntheticSuite
) -> QualityReport:
    """Validate every scored output and compute case-level operator MSE."""
    module = prepared.module
    linear_mse: dict[str, float] = {}
    for case in suite.linear:
        calibration_result = prepared.linear_results[case.name]
        weight_params = calibration_result["weight_params"]
        validate_hif4_params(weight_params, case.weight.shape)
        state = calibration_result["activation_state"]
        errors: list[float] = []
        for activation in case.test:
            activation_params = module.hif4_dynamic_quantize_activation(
                activation.quant, activation.scale, state
            )
            validate_hif4_params(activation_params, activation.shape)
            errors.append(
                linear_operator_mse(
                    activation, case.weight, activation_params, weight_params
                )
            )
        linear_mse[case.name] = statistics.fmean(errors)

    attention_mse: dict[str, float] = {}
    for case in suite.attention:
        states = prepared.attention_states[case.name]
        errors = []
        for q, k, v in case.test:
            qp = module.hif4_dynamic_quantize_q(
                q.quant, q.scale, case.q_num_heads, case.head_dim, states["q_state"]
            )
            kp = module.hif4_dynamic_quantize_k(
                k.quant, k.scale, case.kv_num_heads, case.head_dim, states["k_state"]
            )
            vp = module.hif4_dynamic_quantize_v(
                v.quant, v.scale, case.kv_num_heads, case.head_dim, states["v_state"]
            )
            validate_hif4_params(qp, q.shape)
            validate_hif4_params(kp, k.shape)
            validate_hif4_params(vp, v.shape)
            errors.append(
                attention_operator_mse(
                    (q, k, v),
                    (qp, kp, vp),
                    case.q_num_heads,
                    case.kv_num_heads,
                    case.head_dim,
                )
            )
        attention_mse[case.name] = statistics.fmean(errors)
    return QualityReport(linear_mse, attention_mse)


def compare_solutions(
    baseline_path: str | Path,
    candidate_path: str | Path,
    *,
    seed: int = 20260903,
    profile: str = "default",
    device: str = "cpu",
    repeats: int = 7,
    warmup: int = 2,
) -> ComparisonReport:
    """Load, validate, score, and interleave latency runs of two submissions."""
    generation_start = time.perf_counter_ns()
    suite = build_synthetic_suite(seed=seed, profile=profile, device=device)
    generation_ms = (time.perf_counter_ns() - generation_start) / 1_000_000.0
    baseline_module = _load_solution(baseline_path, "baseline")
    candidate_module = _load_solution(candidate_path, "candidate")

    # Preparation invokes calibration once to obtain dynamic states.  These
    # calls are outside the timer; calibration latency is measured separately
    # in every timed suite repeat.
    baseline_prepared = _prepare_solution(baseline_module, suite)
    candidate_prepared = _prepare_solution(candidate_module, suite)
    baseline_quality = evaluate_quality(baseline_prepared, suite)
    candidate_quality = evaluate_quality(candidate_prepared, suite)
    baseline_timing, candidate_timing = _run_interleaved_timing(
        baseline_prepared,
        candidate_prepared,
        suite,
        repeats,
        warmup,
        seed ^ 0x48494634,
    )
    candidate_total = candidate_timing.overall.total_ms
    speedup = (
        baseline_timing.overall.total_ms / candidate_total
        if candidate_total > 0
        else math.inf
    )
    return ComparisonReport(
        str(Path(baseline_path).resolve()),
        str(Path(candidate_path).resolve()),
        seed,
        profile,
        repeats,
        warmup,
        generation_ms,
        baseline_timing,
        candidate_timing,
        baseline_quality,
        candidate_quality,
        speedup,
    )


def _format_timing_row(label: str, report: TimingReport) -> str:
    overall = report.overall
    return (
        f"{label:<10} {overall.median_ms:>11.4f} {overall.p95_ms:>10.4f} "
        f"{report.median_suite_ms:>13.4f} {report.p95_suite_ms:>12.4f} "
        f"{overall.total_ms:>11.4f}"
    )


def format_report(report: ComparisonReport) -> str:
    """Create a compact human-readable benchmark report."""
    lines = [
        f"HiF4 benchmark profile={report.profile} seed={report.seed} "
        f"repeats={report.repeats} warmup={report.warmup}",
        f"Data generation: {report.data_generation_ms:.3f} ms (excluded from timings)",
        "",
        "Solution    Median call   P95 call  Median suite    P95 suite    Total ms",
        _format_timing_row("baseline", report.baseline_timing),
        _format_timing_row("candidate", report.candidate_timing),
        f"Relative speedup (baseline total / candidate total): "
        f"{report.relative_speedup:.4f}x",
        "",
        "Per-function latency (median / p95 / total ms):",
    ]
    for name in PUBLIC_FUNCTIONS:
        baseline = report.baseline_timing.by_function[name]
        candidate = report.candidate_timing.by_function[name]
        lines.append(
            f"  {name}: baseline {baseline.median_ms:.4f} / "
            f"{baseline.p95_ms:.4f} / {baseline.total_ms:.4f}; candidate "
            f"{candidate.median_ms:.4f} / {candidate.p95_ms:.4f} / "
            f"{candidate.total_ms:.4f}"
        )
    lines.extend(("", "Operator MSE by case (baseline -> candidate):"))
    for name, baseline_mse in report.baseline_quality.linear_mse.items():
        lines.append(
            f"  Linear {name}: {baseline_mse:.8g} -> "
            f"{report.candidate_quality.linear_mse[name]:.8g}"
        )
    for name, baseline_mse in report.baseline_quality.attention_mse.items():
        lines.append(
            f"  Attention {name}: {baseline_mse:.8g} -> "
            f"{report.candidate_quality.attention_mse[name]:.8g}"
        )
    return "\n".join(lines)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="baseline solution.py")
    parser.add_argument("--candidate", required=True, help="candidate solution.py")
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument(
        "--profile", choices=("smoke", "default", "stress"), default="default"
    )
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, ...")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--threads", type=int, help="set torch CPU thread count")
    parser.add_argument("--json", dest="json_path", help="write full JSON report")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.threads is not None:
        if args.threads < 1:
            raise ValueError("--threads must be >= 1")
        torch.set_num_threads(args.threads)
    report = compare_solutions(
        args.baseline,
        args.candidate,
        seed=args.seed,
        profile=args.profile,
        device=args.device,
        repeats=args.repeats,
        warmup=args.warmup,
    )
    print(format_report(report))
    if args.json_path:
        output = Path(args.json_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"JSON report: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
