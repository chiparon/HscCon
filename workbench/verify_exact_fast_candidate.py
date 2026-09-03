"""Random, boundary and timing comparison against solution_reference.py."""

from __future__ import annotations

import statistics
import time

import torch

import solution_exact_fast_candidate as fast
import solution_reference as reference


def dequant(params: dict[str, torch.Tensor]) -> torch.Tensor:
    return (
        params["scale_factor"]
        * params["scale_lv2"]
        * params["scale_lv3"]
        * params["sign"]
        * params["mant"]
    ).flatten(-4)


def compare(q: torch.Tensor, scale: torch.Tensor) -> None:
    dense = reference._nvfp4(q, scale)
    expected = reference._encode(dense)
    actual = fast._quantize_hif4(q, scale)
    expected_x = dequant(expected)
    actual_x = dequant(actual)
    source_x = dense.float()

    # The reduced DP must preserve the exhaustive reference's tie order as well
    # as its reconstruction loss, so all five payload fields are bit-identical.
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    prefix = source_x.shape[:-1]
    grouped_source = source_x.reshape(*prefix, -1, 8, 2, 4)
    grouped_expected = expected_x.reshape_as(grouped_source)
    grouped_actual = actual_x.reshape_as(grouped_source)
    expected_sse = (
        (grouped_source.double() - grouped_expected.double())
        .square()
        .sum(dim=(-1, -2))
    )
    actual_sse = (
        (grouped_source.double() - grouped_actual.double())
        .square()
        .sum(dim=(-1, -2))
    )
    torch.testing.assert_close(actual_sse, expected_sse, rtol=0, atol=0)


def median_ms(fn, q: torch.Tensor, scale: torch.Tensor, repeats: int) -> float:
    for _ in range(2):
        fn(q, scale)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(q, scale)
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def main() -> None:
    torch.manual_seed(19)
    # NVFP4-like half-step carriers with log-wide positive scales.
    for shape in ((1, 64), (3, 256), (2, 5, 512)):
        q = torch.randn(*shape).mul_(2.0).round_().div_(2.0).clamp_(-6, 6)
        scale = torch.pow(
            2.0,
            torch.empty(*shape[:-1], shape[-1] // 16).uniform_(-12, 10),
        )
        compare(q, scale)

    # Wider randomized sweep also exercises exact-error ties where two legal
    # layouts reconstruct different values but attain identical SSE.
    for seed in range(100):
        torch.manual_seed(seed)
        q = torch.randn(4, 1024).mul_(2).round_().div_(2).clamp_(-6, 6)
        scale = torch.pow(2.0, torch.empty(4, 64).uniform_(-20, 12))
        compare(q, scale)

    # Zeros, signs, S1P2 rounding boundaries, saturation, and a very small E6M2
    # scale are all represented in one deterministic set.
    boundary = torch.tensor(
        [
            0.0,
            -0.0,
            0.125,
            -0.125,
            0.375,
            -0.375,
            0.625,
            -0.625,
            1.625,
            -1.625,
            3.5,
            -3.5,
            7.0,
            -7.0,
            14.0,
            -14.0,
        ],
        dtype=torch.float32,
    ).repeat(8)
    compare(boundary.reshape(2, 64), torch.ones(2, 4))
    compare(torch.zeros(2, 64), torch.ones(2, 4))
    compare(boundary.reshape(2, 64), torch.full((2, 4), 2.0**-45))
    print("random and boundary equivalence: OK")

    threads = min(torch.get_num_threads(), 8)
    torch.set_num_threads(threads)
    q = torch.randn(512, 4096).mul_(2.0).round_().div_(2.0).clamp_(-6, 6)
    scale = torch.rand(512, 256).mul_(1.9).add_(0.1)
    ref_ms = median_ms(
        lambda a, b: reference.hif4_dynamic_quantize_activation(a, b, None),
        q,
        scale,
        5,
    )
    fast_ms = median_ms(
        lambda a, b: fast.hif4_dynamic_quantize_activation(a, b, None),
        q,
        scale,
        10,
    )
    print(f"threads={threads}, shape={tuple(q.shape)}")
    print(f"reference median: {ref_ms:.2f} ms")
    print(f"exact DP median:  {fast_ms:.2f} ms")
    print(f"speedup:          {ref_ms / fast_ms:.2f}x")


if __name__ == "__main__":
    main()
