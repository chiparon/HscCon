"""Compare two HiF4 solutions on the public official-layout mini sample.

The tool treats the current solution as the control and reports paired
case-level operator MSE improvements for Linear and both causal/full GQA.
Quantization time is measured separately from operator scoring time.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping

import torch


PUBLIC_FUNCTIONS = (
    "hif4_calibration_and_quantize_weight",
    "hif4_dynamic_quantize_activation",
    "hif4_calibration_attention",
    "hif4_dynamic_quantize_q",
    "hif4_dynamic_quantize_k",
    "hif4_dynamic_quantize_v",
)


def _load_solution(path: str | Path, role: str) -> ModuleType:
    resolved = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(
        f"_hif4_official_{role}_{time.time_ns()}", resolved
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    missing = [name for name in PUBLIC_FUNCTIONS if not callable(getattr(module, name, None))]
    if missing:
        raise AttributeError(f"{resolved} missing {missing}")
    return module


def _nvfp4(pair: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    if isinstance(pair, Mapping):
        quant = pair.get("quant", pair.get("quant_float"))
        scale = pair.get("scale", pair.get("scale_float"))
        if not isinstance(quant, torch.Tensor) or not isinstance(scale, torch.Tensor):
            raise TypeError("NVFP4 dict needs tensor quant/scale fields")
    else:
        quant, scale = pair
    return (
        quant.to(torch.float32)
        .reshape(*quant.shape[:-1], -1, 16)
        .mul(scale.to(torch.float32).unsqueeze(-1))
        .flatten(-2)
        .to(torch.bfloat16)
        .to(torch.float32)
    )


def _hif4(params: Mapping[str, torch.Tensor], shape: torch.Size) -> torch.Tensor:
    value = (
        params["scale_factor"].to(torch.float32)
        * params["scale_lv2"].to(torch.float32)
        * params["scale_lv3"].to(torch.float32)
        * params["sign"].to(torch.float32)
        * params["mant"].to(torch.float32)
    )
    return torch.nan_to_num(value, nan=0.0).reshape(shape)


def _mse(reference: torch.Tensor, actual: torch.Tensor) -> float:
    return float((reference - actual).square().mean().item())


def _nmse(reference: torch.Tensor, actual: torch.Tensor) -> float:
    denominator = reference.square().sum().clamp_min(1.0e-12)
    return float(((reference - actual).square().sum() / denominator).item())


def _gain(control: float, candidate: float) -> float:
    if control == 0.0:
        return 0.0 if candidate == 0.0 else -math.inf
    return 100.0 * (control - candidate) / control


def _attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    causal: bool,
) -> torch.Tensor:
    q_tokens = q.shape[-2]
    kv_tokens = k.shape[-2]
    qh = q.reshape(q_tokens, q_heads, head_dim).transpose(0, 1)
    kh = k.reshape(kv_tokens, kv_heads, head_dim).transpose(0, 1)
    vh = v.reshape(kv_tokens, kv_heads, head_dim).transpose(0, 1)
    if q_heads != kv_heads:
        repeats = q_heads // kv_heads
        kh = kh.repeat_interleave(repeats, dim=0)
        vh = vh.repeat_interleave(repeats, dim=0)
    logits = (qh @ kh.transpose(-1, -2)) * (head_dim**-0.5)
    if causal:
        # Match torch SDPA's is_causal=True upper-left causal bias.  Public
        # samples have equal Q/K lengths, but keeping the rectangular rule
        # explicit prevents a silent bottom-right-mask mismatch.
        row = torch.arange(q_tokens, device=logits.device)[:, None]
        col = torch.arange(kv_tokens, device=logits.device)[None, :]
        allowed = col <= row
        logits.masked_fill_(~allowed, float("-inf"))
    probabilities = torch.softmax(logits, dim=-1)
    return (probabilities @ vh).transpose(0, 1).reshape(q_tokens, q_heads * head_dim)


def _time_call(function, *args) -> tuple[Any, float]:
    started = time.perf_counter_ns()
    result = function(*args)
    return result, (time.perf_counter_ns() - started) / 1_000_000.0


def _state_errors(value: Any, path: str = "state") -> list[str]:
    """Apply the public frozen-state constraints and count container nodes."""
    errors: list[str] = []
    nodes = 0

    def visit(item: Any, name: str, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > 4096:
            errors.append(f"{path}: more than 4096 nodes")
            return
        if depth > 8:
            errors.append(f"{name}: nesting deeper than 8")
            return
        if type(item) is torch.Tensor:
            if item.device.type != "cpu":
                errors.append(f"{name}: state tensor is not on CPU")
            if item.layout is not torch.strided:
                errors.append(f"{name}: state tensor is not dense strided")
            if item.requires_grad:
                errors.append(f"{name}: requires_grad=True")
            if item.is_floating_point() and not torch.isfinite(item).all():
                errors.append(f"{name}: non-finite values")
            return
        if item is None or type(item) in (bool, int, str):
            return
        if type(item) is float:
            if not math.isfinite(item):
                errors.append(f"{name}: non-finite float")
            return
        if type(item) in (list, tuple):
            for index, child in enumerate(item):
                visit(child, f"{name}[{index}]", depth + 1)
            return
        if type(item) is dict:
            for key, child in item.items():
                if type(key) is not str:
                    errors.append(f"{name}: non-string mapping key")
                else:
                    visit(child, f"{name}.{key}", depth + 1)
            return
        errors.append(f"{name}: unsupported type {type(item).__name__}")

    visit(value, path, 0)
    return errors


def _tensor_storage_ids(value: Any) -> set[int]:
    ids: set[int] = set()
    if type(value) is torch.Tensor:
        ids.add(value.untyped_storage().data_ptr())
    elif type(value) in (list, tuple):
        for child in value:
            ids.update(_tensor_storage_ids(child))
    elif type(value) is dict:
        for child in value.values():
            ids.update(_tensor_storage_ids(child))
    return ids


def _validate_params(params: Mapping[str, torch.Tensor], shape: torch.Size) -> list[str]:
    errors: list[str] = []
    channels = int(shape[-1])
    prefix = tuple(shape[:-1]) + (channels // 64,)
    expected = {
        "scale_factor": prefix + (1, 1, 1),
        "scale_lv2": prefix + (8, 1, 1),
        "scale_lv3": prefix + (8, 2, 1),
        "sign": prefix + (8, 2, 4),
        "mant": prefix + (8, 2, 4),
    }
    if not isinstance(params, Mapping):
        return ["params are not a mapping"]
    for key, expected_shape in expected.items():
        value = params.get(key)
        if not isinstance(value, torch.Tensor):
            errors.append(f"{key}: missing/non-tensor")
        elif tuple(value.shape) != expected_shape:
            errors.append(f"{key}: shape {tuple(value.shape)} != {expected_shape}")
        elif not torch.isfinite(value).all():
            errors.append(f"{key}: non-finite")
    if errors:
        return errors
    sf = params["scale_factor"].to(torch.float64)
    exponent = torch.floor(torch.log2(sf.clamp_min(2.0 ** -126)))
    rounded = torch.round(sf * (2.0 ** (2 - exponent))) * (2.0 ** (exponent - 2))
    if not ((sf >= 2.0 ** -48) & (sf <= 49152.0) & (sf == rounded)).all():
        errors.append("scale_factor: illegal E6M2 value")
    if not torch.isin(params["scale_lv2"], torch.tensor([1.0, 2.0])).all():
        errors.append("scale_lv2: illegal value")
    if not torch.isin(params["scale_lv3"], torch.tensor([1.0, 2.0])).all():
        errors.append("scale_lv3: illegal value")
    if not torch.isin(params["sign"], torch.tensor([-1.0, 0.0, 1.0])).all():
        errors.append("sign: illegal value")
    mant = params["mant"]
    if not ((mant >= 0) & (mant <= 1.75) & (mant * 4 == torch.round(mant * 4))).all():
        errors.append("mant: illegal value")
    return errors


def _linear_result(module: ModuleType, group: Mapping[str, Any]) -> dict[str, Any]:
    calibration, calibration_ms = _time_call(
        module.hif4_calibration_and_quantize_weight,
        *group["weight"],
        group["calib_activation_list"],
    )
    source_weight = _nvfp4(group["weight"])
    converted_weight = _hif4(calibration["weight_params"], source_weight.shape)
    validation_errors = _validate_params(calibration["weight_params"], source_weight.shape)
    validation_errors.extend(_state_errors(calibration["activation_state"], "activation_state"))
    calibration_storage = _tensor_storage_ids(group["calib_activation_list"])
    state_storage = _tensor_storage_ids(calibration["activation_state"])
    if calibration_storage & state_storage:
        validation_errors.append("activation_state aliases a calibration input tensor")
    cases = []
    dynamic_ms = 0.0
    for index, pair in enumerate(group["test_activation_list"]):
        params, elapsed = _time_call(
            module.hif4_dynamic_quantize_activation,
            *pair,
            calibration["activation_state"],
        )
        dynamic_ms += elapsed
        source_activation = _nvfp4(pair)
        validation_errors.extend(
            _validate_params(params, source_activation.shape)
        )
        converted_activation = _hif4(params, source_activation.shape)
        reference = source_activation @ source_weight.transpose(-1, -2)
        actual = converted_activation @ converted_weight.transpose(-1, -2)
        cases.append(
            {
                "case": index,
                "mse": _mse(reference, actual),
                "nmse": _nmse(reference, actual),
            }
        )
    return {
        "calibration_ms": calibration_ms,
        "dynamic_ms": dynamic_ms,
        "quantization_total_ms": calibration_ms + dynamic_ms,
        "cases": cases,
        "validation_errors": validation_errors,
    }


def _attention_result(module: ModuleType, group: Mapping[str, Any]) -> dict[str, Any]:
    q_heads = int(group["q_num_heads"])
    kv_heads = int(group["kv_num_heads"])
    head_dim = int(group["head_dim"])
    state, calibration_ms = _time_call(
        module.hif4_calibration_attention,
        group["calib"],
        q_heads,
        kv_heads,
        head_dim,
    )
    cases = []
    validation_errors = []
    for role in ("q", "k", "v"):
        validation_errors.extend(_state_errors(state[f"{role}_state"], f"{role}_state"))
    calibration_storage = _tensor_storage_ids(group["calib"])
    for role in ("q", "k", "v"):
        if calibration_storage & _tensor_storage_ids(state[f"{role}_state"]):
            validation_errors.append(f"{role}_state aliases a calibration input tensor")
    dynamic_ms = 0.0
    calls = (
        ("q", module.hif4_dynamic_quantize_q, q_heads),
        ("k", module.hif4_dynamic_quantize_k, kv_heads),
        ("v", module.hif4_dynamic_quantize_v, kv_heads),
    )
    for index, sample in enumerate(group["test"]):
        source: dict[str, torch.Tensor] = {}
        converted: dict[str, torch.Tensor] = {}
        for role, function, heads in calls:
            pair = sample[role]
            params, elapsed = _time_call(
                function,
                *pair,
                heads,
                head_dim,
                state[f"{role}_state"],
            )
            dynamic_ms += elapsed
            source[role] = _nvfp4(pair)
            validation_errors.extend(_validate_params(params, source[role].shape))
            converted[role] = _hif4(params, source[role].shape)

        metrics: dict[str, float | int] = {"case": index}
        for causal, label in ((False, "full"), (True, "causal")):
            reference = _attention(
                source["q"], source["k"], source["v"],
                q_heads, kv_heads, head_dim, causal,
            )
            actual = _attention(
                converted["q"], converted["k"], converted["v"],
                q_heads, kv_heads, head_dim, causal,
            )
            metrics[f"{label}_mse"] = _mse(reference, actual)
            metrics[f"{label}_nmse"] = _nmse(reference, actual)
        cases.append(metrics)
    return {
        "calibration_ms": calibration_ms,
        "dynamic_ms": dynamic_ms,
        "quantization_total_ms": calibration_ms + dynamic_ms,
        "cases": cases,
        "validation_errors": validation_errors,
    }


def _timing_invocations(
    module: ModuleType, linear: Mapping[str, Any], attention: Mapping[str, Any]
) -> dict[str, Any]:
    linear_state = module.hif4_calibration_and_quantize_weight(
        *linear["weight"], linear["calib_activation_list"]
    )["activation_state"]
    q_heads = int(attention["q_num_heads"])
    kv_heads = int(attention["kv_num_heads"])
    head_dim = int(attention["head_dim"])
    attention_state = module.hif4_calibration_attention(
        attention["calib"], q_heads, kv_heads, head_dim
    )
    calls: dict[str, Any] = {
        "linear/calibration": lambda: module.hif4_calibration_and_quantize_weight(
            *linear["weight"], linear["calib_activation_list"]
        ),
        "attention/calibration": lambda: module.hif4_calibration_attention(
            attention["calib"], q_heads, kv_heads, head_dim
        ),
    }
    for index, pair in enumerate(linear["test_activation_list"]):
        calls[f"linear/activation/{index}"] = (
            lambda pair=pair: module.hif4_dynamic_quantize_activation(
                *pair, linear_state
            )
        )
    for index, sample in enumerate(attention["test"]):
        calls[f"attention/q/{index}"] = (
            lambda sample=sample: module.hif4_dynamic_quantize_q(
                *sample["q"], q_heads, head_dim, attention_state["q_state"]
            )
        )
        calls[f"attention/k/{index}"] = (
            lambda sample=sample: module.hif4_dynamic_quantize_k(
                *sample["k"], kv_heads, head_dim, attention_state["k_state"]
            )
        )
        calls[f"attention/v/{index}"] = (
            lambda sample=sample: module.hif4_dynamic_quantize_v(
                *sample["v"], kv_heads, head_dim, attention_state["v_state"]
            )
        )
    return calls


def _interleaved_timing(
    control_module: ModuleType,
    candidate_module: ModuleType,
    linear: Mapping[str, Any],
    attention: Mapping[str, Any],
    warmups: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    modules = {"control": control_module, "candidate": candidate_module}
    calls = {
        label: _timing_invocations(module, linear, attention)
        for label, module in modules.items()
    }
    keys = list(calls["control"])
    if set(keys) != set(calls["candidate"]):
        raise AssertionError("control/candidate timing call sets differ")

    for _ in range(warmups):
        for key in keys:
            for label in ("control", "candidate"):
                calls[label][key]()

    rng = random.Random(seed)
    durations = {
        label: {key: [] for key in keys} for label in ("control", "candidate")
    }
    suite_totals = {"control": [], "candidate": []}
    for _ in range(repeats):
        run_keys = list(keys)
        rng.shuffle(run_keys)
        totals = {"control": 0.0, "candidate": 0.0}
        for key in run_keys:
            order = ["control", "candidate"]
            rng.shuffle(order)
            for label in order:
                _, elapsed = _time_call(calls[label][key])
                durations[label][key].append(elapsed)
                totals[label] += elapsed
        for label in totals:
            suite_totals[label].append(totals[label])

    report: dict[str, Any] = {}
    for label in ("control", "candidate"):
        by_stage: dict[str, list[float]] = {
            "linear_calibration": [],
            "linear_dynamic": [],
            "attention_calibration": [],
            "attention_dynamic": [],
        }
        for key, values in durations[label].items():
            if key == "linear/calibration":
                stage = "linear_calibration"
            elif key.startswith("linear/"):
                stage = "linear_dynamic"
            elif key == "attention/calibration":
                stage = "attention_calibration"
            else:
                stage = "attention_dynamic"
            by_stage[stage].extend(values)
        report[label] = {
            "median_suite_ms": statistics.median(suite_totals[label]),
            "suite_ms": suite_totals[label],
            "stage_call_median_ms": {
                stage: statistics.median(values) for stage, values in by_stage.items()
            },
        }
    report["runtime_ratio"] = (
        report["candidate"]["median_suite_ms"]
        / report["control"]["median_suite_ms"]
    )
    report["seed"] = seed
    report["warmups"] = warmups
    report["repeats"] = repeats
    report["randomized_paired_interleaving"] = True
    return report


def _comparison(
    control: dict[str, Any], candidate: dict[str, Any], timing: dict[str, Any]
) -> dict[str, Any]:
    linear_gains = [
        _gain(a["mse"], b["mse"])
        for a, b in zip(control["linear"]["cases"], candidate["linear"]["cases"])
    ]
    linear_nmse_gains = [
        _gain(a["nmse"], b["nmse"])
        for a, b in zip(control["linear"]["cases"], candidate["linear"]["cases"])
    ]
    attention_full_gains = [
        _gain(a["full_mse"], b["full_mse"])
        for a, b in zip(control["attention"]["cases"], candidate["attention"]["cases"])
    ]
    attention_full_nmse_gains = [
        _gain(a["full_nmse"], b["full_nmse"])
        for a, b in zip(control["attention"]["cases"], candidate["attention"]["cases"])
    ]
    attention_causal_gains = [
        _gain(a["causal_mse"], b["causal_mse"])
        for a, b in zip(control["attention"]["cases"], candidate["attention"]["cases"])
    ]
    attention_causal_nmse_gains = [
        _gain(a["causal_nmse"], b["causal_nmse"])
        for a, b in zip(control["attention"]["cases"], candidate["attention"]["cases"])
    ]
    all_gains = linear_gains + attention_full_gains + attention_causal_gains
    paired_gain_deltas = [
        abs(a - b)
        for a, b in zip(
            linear_gains + attention_full_gains + attention_causal_gains,
            linear_nmse_gains
            + attention_full_nmse_gains
            + attention_causal_nmse_gains,
        )
    ]
    validation_errors = (
        control["linear"]["validation_errors"]
        + control["attention"]["validation_errors"]
        + candidate["linear"]["validation_errors"]
        + candidate["attention"]["validation_errors"]
    )
    return {
        "linear_case_gain_percent": linear_gains,
        "linear_macro_gain_percent": statistics.fmean(linear_gains),
        "attention_full_case_gain_percent": attention_full_gains,
        "attention_full_macro_gain_percent": statistics.fmean(attention_full_gains),
        "attention_causal_case_gain_percent": attention_causal_gains,
        "attention_causal_macro_gain_percent": statistics.fmean(attention_causal_gains),
        "overall_macro_gain_percent": statistics.fmean(all_gains),
        "worst_case_gain_percent": min(all_gains),
        "max_mse_nmse_gain_delta_percent": max(paired_gain_deltas),
        "mse_nmse_gain_consistent": max(paired_gain_deltas) < 1.0e-4,
        "runtime_ratio": timing["runtime_ratio"],
        "control_quantization_ms": timing["control"]["median_suite_ms"],
        "candidate_quantization_ms": timing["candidate"]["median_suite_ms"],
        "validation_errors": validation_errors,
        "passes_legality_and_state_gate": not validation_errors,
        "passes_five_percent_accuracy_gate": statistics.fmean(all_gains) > 5.0,
    }


def compare(
    control_path: str | Path,
    candidate_path: str | Path,
    dataset_dir: str | Path,
    warmups: int = 1,
    repeats: int = 3,
    seed: int = 20260903,
) -> dict[str, Any]:
    directory = Path(dataset_dir)
    linear_data = torch.load(directory / "linear.pt", map_location="cpu", weights_only=True)
    attention_data = torch.load(directory / "attn.pt", map_location="cpu", weights_only=True)
    if len(linear_data) != 1 or len(attention_data) != 1:
        raise ValueError("public mini sample is expected to contain one group per problem")
    control_module = _load_solution(control_path, "control")
    candidate_module = _load_solution(candidate_path, "candidate")
    control = {
        "linear": _linear_result(control_module, linear_data[0]),
        "attention": _attention_result(control_module, attention_data[0]),
    }
    candidate = {
        "linear": _linear_result(candidate_module, linear_data[0]),
        "attention": _attention_result(candidate_module, attention_data[0]),
    }
    timing = _interleaved_timing(
        control_module, candidate_module, linear_data[0], attention_data[0],
        warmups, repeats, seed,
    )
    return {
        "control_path": str(Path(control_path).resolve()),
        "candidate_path": str(Path(candidate_path).resolve()),
        "dataset_dir": str(directory.resolve()),
        "control": control,
        "candidate": candidate,
        "timing": timing,
        "audit": {
            "calibration_received_test_data": False,
            "quality_uses_linear_a_times_w_transpose": True,
            "attention_layout": "[tokens, heads * head_dim]; KV heads repeat_interleave for GQA",
            "causal_mask": "upper-left, key_position <= query_position",
            "source_dequantization": "FP32 multiply rounded through BF16",
        },
        "comparison": _comparison(control, candidate, timing),
    }


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--json")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    report = compare(
        args.control, args.candidate, args.dataset_dir,
        warmups=args.warmups, repeats=args.repeats, seed=args.seed,
    )
    print(json.dumps(report["comparison"], indent=2, sort_keys=True))
    if args.json:
        output = Path(args.json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"JSON report: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
