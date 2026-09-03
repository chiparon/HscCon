"""Minimal contract/value-domain smoke test for solution_fast_candidate.py."""

import torch

import solution_fast_candidate as solution


def check(params: solution.HiF4Params, prefix: tuple[int, ...], channels: int) -> None:
    blocks = channels // 64
    assert params["scale_factor"].shape == prefix + (blocks, 1, 1, 1)
    assert params["scale_lv2"].shape == prefix + (blocks, 8, 1, 1)
    assert params["scale_lv3"].shape == prefix + (blocks, 8, 2, 1)
    assert params["sign"].shape == prefix + (blocks, 8, 2, 4)
    assert params["mant"].shape == prefix + (blocks, 8, 2, 4)
    assert torch.isfinite(params["scale_factor"]).all()
    assert (params["scale_factor"] > 0).all()
    assert torch.isin(params["scale_lv2"], torch.tensor([1.0, 2.0])).all()
    assert torch.isin(params["scale_lv3"], torch.tensor([1.0, 2.0])).all()
    assert torch.isin(params["sign"], torch.tensor([-1.0, 0.0, 1.0])).all()
    assert torch.isin(params["mant"], torch.arange(8) * 0.25).all()


def main() -> None:
    torch.manual_seed(7)
    prefix, channels = (3, 5), 256
    q = torch.randn(*prefix, channels).mul_(2).round().div_(2)
    s = torch.rand(*prefix, channels // 16).mul_(1.9).add_(0.1)

    linear = solution.hif4_calibration_and_quantize_weight(q, s, [])
    check(linear["weight_params"], prefix, channels)
    check(
        solution.hif4_dynamic_quantize_activation(
            q, s, linear["activation_state"]
        ),
        prefix,
        channels,
    )

    states = solution.hif4_calibration_attention([], 4, 2, 64)
    check(solution.hif4_dynamic_quantize_q(q, s, 4, 64, states["q_state"]), prefix, channels)
    check(solution.hif4_dynamic_quantize_k(q, s, 2, 64, states["k_state"]), prefix, channels)
    check(solution.hif4_dynamic_quantize_v(q, s, 2, 64, states["v_state"]), prefix, channels)

    zq = torch.zeros(1, 64)
    zs = torch.ones(1, 4)
    zero = solution.hif4_dynamic_quantize_activation(zq, zs, {})
    check(zero, (1,), 64)
    assert (zero["scale_factor"] == 2.0**-48).all()
    assert (zero["mant"] == 0).all()
    print("fast candidate smoke: OK")


if __name__ == "__main__":
    main()
