from pathlib import Path

import pytest
import torch

from tools import hif4_benchmark as bench


FIXTURE_SOLUTION = Path(__file__).with_name("benchmark_fixture_solution.py")


def _constant_params(shape: tuple[int, ...]) -> dict[str, torch.Tensor]:
    prefix = shape[:-1]
    blocks = shape[-1] // 64
    return {
        "scale_factor": torch.ones(prefix + (blocks, 1, 1, 1)),
        "scale_lv2": torch.ones(prefix + (blocks, 8, 1, 1)),
        "scale_lv3": torch.ones(prefix + (blocks, 8, 2, 1)),
        "sign": torch.ones(prefix + (blocks, 8, 2, 4)),
        "mant": torch.full(prefix + (blocks, 8, 2, 4), 0.5),
    }


def test_synthetic_suite_is_deterministic_and_covers_gqa_outliers() -> None:
    first = bench.build_synthetic_suite(seed=17, profile="smoke")
    second = bench.build_synthetic_suite(seed=17, profile="smoke")

    torch.testing.assert_close(first.linear[1].weight.quant, second.linear[1].weight.quant)
    torch.testing.assert_close(first.linear[1].weight.scale, second.linear[1].weight.scale)
    assert any(case.q_num_heads > case.kv_num_heads for case in first.attention)
    assert any("outlier" in case.name for case in first.linear)
    assert any("outlier" in case.name for case in first.attention)
    assert {case.weight.shape[-1] for case in first.linear} == {64, 128}


def test_nvfp4_quantization_uses_legal_carrier_levels_and_round_trips() -> None:
    values = torch.linspace(-20, 20, 128).reshape(2, 64)
    item = bench.quantize_synthetic_nvfp4(values)
    expected_scale_shape = (2, 4)
    assert item.scale.shape == expected_scale_shape
    legal_magnitudes = torch.isin(item.quant.abs(), bench.NVFP4_LEVELS)
    assert bool(legal_magnitudes.all())
    assert bench.dequantize_nvfp4(item).shape == values.shape


def test_e6m2_legality_matches_bias_range_and_reserved_nan() -> None:
    legal = torch.tensor(
        [2.0**-48, 1.25 * 2.0**-20, 1.75, 1.5 * 2.0**15, float("nan")]
    )
    illegal = torch.tensor([0.0, -1.0, 1.1, 2.0**-49, 1.75 * 2.0**15])
    assert bool(bench.e6m2_legal_mask(legal).all())
    assert not bool(bench.e6m2_legal_mask(illegal).any())
    assert not bool(bench.e6m2_legal_mask(torch.tensor(float("nan")), False))


def test_hif4_validation_and_dequantization() -> None:
    shape = (2, 64)
    params = _constant_params(shape)
    params["scale_lv2"][:, :, :4] = 2.0
    params["scale_lv3"][:, :, :, :1] = 2.0
    bench.validate_hif4_params(params, shape)
    restored = bench.dequantize_hif4(params, shape)
    assert restored.shape == shape
    assert torch.isfinite(restored).all()

    params["mant"][0, 0, 0, 0] = 0.3
    with pytest.raises(bench.HiF4ValidationError, match="mant"):
        bench.validate_hif4_params(params, shape)


def test_hif4_nan_scale_is_zero_block_sentinel() -> None:
    params = _constant_params((64,))
    params["scale_factor"].fill_(float("nan"))
    restored = bench.dequantize_hif4(params, (64,))
    torch.testing.assert_close(restored, torch.zeros(64))


def test_attention_operator_mse_supports_gqa() -> None:
    suite = bench.build_synthetic_suite(seed=23, profile="smoke")
    case = next(case for case in suite.attention if case.q_num_heads > case.kv_num_heads)
    q, k, v = case.test[0]
    module = bench._load_solution(FIXTURE_SOLUTION, "fixture_gqa")
    qp = module.hif4_dynamic_quantize_q(
        q.quant, q.scale, case.q_num_heads, case.head_dim, None
    )
    kp = module.hif4_dynamic_quantize_k(
        k.quant, k.scale, case.kv_num_heads, case.head_dim, None
    )
    vp = module.hif4_dynamic_quantize_v(
        v.quant, v.scale, case.kv_num_heads, case.head_dim, None
    )
    mse = bench.attention_operator_mse(
        (q, k, v),
        (qp, kp, vp),
        case.q_num_heads,
        case.kv_num_heads,
        case.head_dim,
    )
    assert mse >= 0.0
    assert torch.isfinite(torch.tensor(mse))


def test_compare_solutions_times_all_six_interfaces() -> None:
    report = bench.compare_solutions(
        FIXTURE_SOLUTION,
        FIXTURE_SOLUTION,
        seed=29,
        profile="smoke",
        repeats=2,
        warmup=1,
    )
    assert set(report.baseline_timing.by_function) == set(bench.PUBLIC_FUNCTIONS)
    assert set(report.candidate_timing.by_function) == set(bench.PUBLIC_FUNCTIONS)
    assert report.baseline_timing.overall.calls > 0
    assert report.candidate_timing.overall.calls == report.baseline_timing.overall.calls
    assert report.baseline_timing.overall.total_ms > 0
    assert report.candidate_timing.overall.total_ms > 0
    assert report.relative_speedup > 0
    rendered = bench.format_report(report)
    assert "Median call" in rendered
    assert "P95 call" in rendered
    assert "Relative speedup" in rendered
    assert "Operator MSE" in rendered
