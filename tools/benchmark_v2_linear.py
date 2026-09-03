"""Focused Linear operator benchmark for calibration-aware HiF4 candidates.

The runner accepts the public ``linear.pt`` payload directly.  Quantization
latency excludes data loading and operator-MSE evaluation.  Baseline and
candidate calls are warmed up, shuffled, and interleaved with a fixed seed.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import random
import statistics
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import torch


def _load_module(path: str, role: str) -> ModuleType:
    resolved = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(f"linear_{role}", resolved)
    if spec is None or spec.loader is None:
        raise ImportError(resolved)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dequant_nvfp4(quant: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # Match the conversion contract: the NVFP4 carrier/scale product is first
    # materialized as BF16, then promoted for operator evaluation.
    blocked = quant.reshape(*quant.shape[:-1], -1, 16)
    return (
        (blocked * scale.unsqueeze(-1))
        .flatten(-2)
        .to(torch.bfloat16)
        .to(torch.float32)
    )


def _dequant_hif4(params: Mapping[str, torch.Tensor], shape: torch.Size) -> torch.Tensor:
    return (
        params["sign"].to(torch.float32)
        * params["mant"].to(torch.float32)
        * params["scale_lv3"].to(torch.float32)
        * params["scale_lv2"].to(torch.float32)
        * params["scale_factor"].to(torch.float32)
    ).reshape(shape)


def _prepare(module: ModuleType, groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        module.hif4_calibration_and_quantize_weight(
            group["weight"][0],
            group["weight"][1],
            group["calib_activation_list"],
        )
        for group in groups
    ]


def _quality(
    module: ModuleType,
    groups: list[dict[str, Any]],
    prepared: list[dict[str, Any]],
) -> list[float]:
    errors: list[float] = []
    for group, result in zip(groups, prepared):
        source_w = _dequant_nvfp4(*group["weight"])
        converted_w = _dequant_hif4(result["weight_params"], source_w.shape)
        for activation in group["test_activation_list"]:
            source_a = _dequant_nvfp4(*activation)
            converted_params = module.hif4_dynamic_quantize_activation(
                activation[0], activation[1], result["activation_state"]
            )
            converted_a = _dequant_hif4(converted_params, source_a.shape)
            reference = source_a @ source_w.transpose(-1, -2)
            actual = converted_a @ converted_w.transpose(-1, -2)
            errors.append(float((reference - actual).square().mean().item()))
    return errors


def _invocations(
    module: ModuleType,
    groups: list[dict[str, Any]],
    prepared: list[dict[str, Any]],
) -> list[tuple[str, Any]]:
    calls: list[tuple[str, Any]] = []
    for group, result in zip(groups, prepared):
        calls.append(
            (
                "calibration",
                lambda group=group: module.hif4_calibration_and_quantize_weight(
                    group["weight"][0],
                    group["weight"][1],
                    group["calib_activation_list"],
                ),
            )
        )
        for activation in group["test_activation_list"]:
            calls.append(
                (
                    "dynamic",
                    lambda activation=activation, state=result["activation_state"]: (
                        module.hif4_dynamic_quantize_activation(
                            activation[0], activation[1], state
                        )
                    ),
                )
            )
    return calls


def _timing(
    modules: tuple[ModuleType, ModuleType],
    groups: list[dict[str, Any]],
    prepared: tuple[list[dict[str, Any]], list[dict[str, Any]]],
    repeats: int,
    warmup: int,
    seed: int,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    calls = tuple(
        _invocations(module, groups, state)
        for module, state in zip(modules, prepared)
    )
    rng = random.Random(seed)
    indices = list(range(len(calls[0])))
    for warmup_index in range(warmup):
        rng.shuffle(indices)
        for index in indices:
            order = (0, 1) if (index + warmup_index) % 2 == 0 else (1, 0)
            for solution_index in order:
                calls[solution_index][index][1]()
    records = ({"calibration": [], "dynamic": [], "suite": []},
               {"calibration": [], "dynamic": [], "suite": []})
    enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            rng.shuffle(indices)
            suite_totals = [0.0, 0.0]
            for index in indices:
                order = [0, 1]
                rng.shuffle(order)
                for solution_index in order:
                    label, call = calls[solution_index][index]
                    start = time.perf_counter_ns()
                    call()
                    elapsed = (time.perf_counter_ns() - start) / 1.0e6
                    records[solution_index][label].append(elapsed)
                    suite_totals[solution_index] += elapsed
            for solution_index in (0, 1):
                records[solution_index]["suite"].append(suite_totals[solution_index])
    finally:
        if enabled:
            gc.enable()
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--candidate-weight-offsets",
        help="comma-separated E6M2 neighbor offsets for candidate experiments",
    )
    parser.add_argument(
        "--candidate-activation-offsets",
        help="comma-separated E6M2 neighbor offsets for candidate experiments",
    )
    parser.add_argument("--disable-weight-importance", action="store_true")
    parser.add_argument("--disable-activation-importance", action="store_true")
    parser.add_argument("--importance-min", type=float)
    parser.add_argument("--importance-max", type=float)
    parser.add_argument("--weight-importance-power", type=float)
    parser.add_argument("--activation-importance-power", type=float)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    groups = torch.load(args.data, weights_only=True, map_location="cpu")
    modules = (
        _load_module(args.baseline, "baseline"),
        _load_module(args.candidate, "candidate"),
    )
    if args.candidate_weight_offsets:
        modules[1]._WEIGHT_BASE_OFFSETS = tuple(
            int(value) for value in args.candidate_weight_offsets.split(",")
        )
    if args.candidate_activation_offsets:
        modules[1]._ACTIVATION_BASE_OFFSETS = tuple(
            int(value) for value in args.candidate_activation_offsets.split(",")
        )
    if args.disable_weight_importance:
        modules[1]._USE_WEIGHT_IMPORTANCE = False
    if args.disable_activation_importance:
        modules[1]._USE_ACTIVATION_IMPORTANCE = False
    if args.importance_min is not None:
        modules[1]._IMPORTANCE_MIN = args.importance_min
    if args.importance_max is not None:
        modules[1]._IMPORTANCE_MAX = args.importance_max
    if args.weight_importance_power is not None:
        modules[1]._WEIGHT_IMPORTANCE_POWER = args.weight_importance_power
    if args.activation_importance_power is not None:
        modules[1]._ACTIVATION_IMPORTANCE_POWER = args.activation_importance_power
    prepared = (_prepare(modules[0], groups), _prepare(modules[1], groups))
    quality = (
        _quality(modules[0], groups, prepared[0]),
        _quality(modules[1], groups, prepared[1]),
    )
    improvements = [
        100.0 * (before - after) / before
        for before, after in zip(quality[0], quality[1])
    ]
    print("Linear operator MSE (baseline -> candidate, improvement):")
    for index, (before, after, improvement) in enumerate(
        zip(quality[0], quality[1], improvements)
    ):
        print(f"  test {index}: {before:.9g} -> {after:.9g} ({improvement:+.3f}%)")
    aggregate = 100.0 * (sum(quality[0]) - sum(quality[1])) / sum(quality[0])
    wins = sum(value > 5.0 for value in improvements)
    print(f"aggregate improvement: {aggregate:+.3f}%")
    print(f">5% cases: {wins}/{len(improvements)}")

    timing = _timing(
        modules, groups, prepared, args.repeats, args.warmup, args.seed
    )
    print("Quantizer latency median ms (baseline -> candidate):")
    for label in ("calibration", "dynamic", "suite"):
        before = statistics.median(timing[0][label])
        after = statistics.median(timing[1][label])
        print(f"  {label}: {before:.3f} -> {after:.3f} ({after / before:.3f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
