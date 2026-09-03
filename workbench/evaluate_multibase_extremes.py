"""Deterministic extreme-distribution audit for guarded multibase search."""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import hif4_benchmark as hb


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _extreme_values(
    shape: tuple[int, ...],
    generator: torch.Generator,
    *,
    sparse: float,
    exponent_low: float,
    exponent_high: float,
    outlier_stride: int,
) -> torch.Tensor:
    exponent = exponent_low + (exponent_high - exponent_low) * torch.rand(
        shape, generator=generator
    )
    sign = torch.where(torch.rand(shape, generator=generator) < 0.5, -1.0, 1.0)
    values = sign * torch.exp2(exponent)
    values.masked_fill_(torch.rand(shape, generator=generator) < sparse, 0.0)
    flat = values.reshape(-1, shape[-1])
    if outlier_stride > 0:
        flat[:, ::outlier_stride] = torch.exp2(
            torch.tensor(exponent_high + 2.0)
        )
    return values


def _nv(
    shape: tuple[int, ...],
    generator: torch.Generator,
    **kwargs,
) -> hb.NVFP4Tensor:
    return hb.quantize_synthetic_nvfp4(
        _extreme_values(shape, generator, **kwargs)
    )


def build_extreme_suite(seed: int = 0xE6_42) -> hb.SyntheticSuite:
    g = torch.Generator().manual_seed(seed)
    linear = (
        hb.LinearCase(
            "linear_loguniform_sparse",
            _nv(
                (192, 512), g, sparse=0.65, exponent_low=-8, exponent_high=7,
                outlier_stride=64,
            ),
            tuple(
                _nv(
                    (40, 512), g, sparse=0.55, exponent_low=-7,
                    exponent_high=6, outlier_stride=128,
                )
                for _ in range(2)
            ),
            tuple(
                _nv(
                    (43 + i * 4, 512), g, sparse=0.58, exponent_low=-8,
                    exponent_high=7, outlier_stride=128,
                )
                for i in range(2)
            ),
        ),
        hb.LinearCase(
            "linear_single_outlier_blocks",
            _nv(
                (128, 256), g, sparse=0.15, exponent_low=-10, exponent_high=1,
                outlier_stride=64,
            ),
            tuple(
                _nv(
                    (32, 256), g, sparse=0.10, exponent_low=-9,
                    exponent_high=2, outlier_stride=64,
                )
                for _ in range(2)
            ),
            tuple(
                _nv(
                    (35 + i * 3, 256), g, sparse=0.10, exponent_low=-9,
                    exponent_high=2, outlier_stride=64,
                )
                for i in range(2)
            ),
        ),
    )

    def qkv(
        batch: int,
        qt: int,
        kt: int,
        qh: int,
        kvh: int,
        hd: int,
        extra: int,
    ) -> tuple[hb.NVFP4Tensor, hb.NVFP4Tensor, hb.NVFP4Tensor]:
        q = _nv(
            (batch, qt + extra, qh * hd), g, sparse=0.35, exponent_low=-8,
            exponent_high=5, outlier_stride=hd,
        )
        k = _nv(
            (batch, kt + extra, kvh * hd), g, sparse=0.30, exponent_low=-8,
            exponent_high=5, outlier_stride=hd,
        )
        v = _nv(
            (batch, kt + extra, kvh * hd), g, sparse=0.70, exponent_low=-10,
            exponent_high=6, outlier_stride=64,
        )
        return q, k, v

    attention = (
        hb.AttentionCase(
            "attention_gqa_head_skew",
            16,
            2,
            64,
            tuple(qkv(1, 24, 40, 16, 2, 64, i) for i in range(2)),
            tuple(qkv(1, 27, 43, 16, 2, 64, 2 + i) for i in range(2)),
        ),
        hb.AttentionCase(
            "attention_mha_sparse_value",
            8,
            8,
            64,
            tuple(qkv(1, 20, 28, 8, 8, 64, i) for i in range(2)),
            tuple(qkv(1, 23, 31, 8, 8, 64, 2 + i) for i in range(2)),
        ),
    )
    return hb.SyntheticSuite(seed, "extreme", linear, attention)


def _tensor_sse(item: hb.NVFP4Tensor, params) -> tuple[float, int]:
    source = hb.dequantize_nvfp4(item)
    actual = hb.dequantize_hif4(params, item.shape)
    return float((source - actual).square().sum().item()), source.numel()


def _offsets(base_params, candidate_params) -> Counter[int]:
    before = base_params["scale_factor"].to(torch.float32).contiguous().view(torch.int32)
    after = candidate_params["scale_factor"].to(torch.float32).contiguous().view(torch.int32)
    delta = ((after - before) // 0x00200000).reshape(-1).tolist()
    return Counter(int(x) for x in delta)


def main() -> None:
    torch.set_num_threads(8)
    suite = build_extreme_suite()
    base_module = _load(ROOT / "solution.py", "extreme_base")
    candidate_module = _load(
        ROOT / "workbench" / "solution_v2_multibase_candidate.py",
        "extreme_candidate",
    )
    base = hb._prepare_solution(base_module, suite)
    candidate = hb._prepare_solution(candidate_module, suite)
    base_quality = hb.evaluate_quality(base, suite)
    candidate_quality = hb.evaluate_quality(candidate, suite)

    print("Extreme operator MSE (V1 -> guarded 3-base)")
    for name, mse in base_quality.linear_mse.items():
        other = candidate_quality.linear_mse[name]
        print(f"Linear {name}: {mse:.9g} -> {other:.9g}; gain={(mse-other)/mse:+.3%}")
    for name, mse in base_quality.attention_mse.items():
        other = candidate_quality.attention_mse[name]
        print(f"Attention {name}: {mse:.9g} -> {other:.9g}; gain={(mse-other)/mse:+.3%}")

    sse = {"base": defaultdict(float), "candidate": defaultdict(float)}
    counts = defaultdict(int)
    offsets: dict[str, Counter[int]] = defaultdict(Counter)
    for case in suite.linear:
        bp = base.linear_results[case.name]["weight_params"]
        cp = candidate.linear_results[case.name]["weight_params"]
        for label, params in (("base", bp), ("candidate", cp)):
            value, count = _tensor_sse(case.weight, params)
            sse[label]["w"] += value
            if label == "base":
                counts["w"] += count
        offsets["w"].update(_offsets(bp, cp))
        for item in case.test:
            bap = base_module.hif4_dynamic_quantize_activation(
                item.quant, item.scale, base.linear_results[case.name]["activation_state"]
            )
            cap = candidate_module.hif4_dynamic_quantize_activation(
                item.quant,
                item.scale,
                candidate.linear_results[case.name]["activation_state"],
            )
            for label, params in (("base", bap), ("candidate", cap)):
                value, count = _tensor_sse(item, params)
                sse[label]["a"] += value
                if label == "base":
                    counts["a"] += count
            offsets["a"].update(_offsets(bap, cap))

    for case in suite.attention:
        bs = base.attention_states[case.name]
        cs = candidate.attention_states[case.name]
        funcs = (
            ("q", base_module.hif4_dynamic_quantize_q, candidate_module.hif4_dynamic_quantize_q, case.q_num_heads),
            ("k", base_module.hif4_dynamic_quantize_k, candidate_module.hif4_dynamic_quantize_k, case.kv_num_heads),
            ("v", base_module.hif4_dynamic_quantize_v, candidate_module.hif4_dynamic_quantize_v, case.kv_num_heads),
        )
        for triplet in case.test:
            for index, (role, bf, cf, heads) in enumerate(funcs):
                item = triplet[index]
                bp = bf(item.quant, item.scale, heads, case.head_dim, bs[f"{role}_state"])
                cp = cf(item.quant, item.scale, heads, case.head_dim, cs[f"{role}_state"])
                for label, params in (("base", bp), ("candidate", cp)):
                    value, count = _tensor_sse(item, params)
                    sse[label][role] += value
                    if label == "base":
                        counts[role] += count
                offsets[role].update(_offsets(bp, cp))

    print("Tensor MSE and selected E6M2-code offsets")
    for role in ("w", "a", "q", "k", "v"):
        before = sse["base"][role] / counts[role]
        after = sse["candidate"][role] / counts[role]
        print(
            f"{role}: {before:.9g} -> {after:.9g}; "
            f"gain={(before-after)/before:+.3%}; offsets={dict(sorted(offsets[role].items()))}"
        )


if __name__ == "__main__":
    main()
