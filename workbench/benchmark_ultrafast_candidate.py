"""Interleaved benchmark of the first-round and allocation-reduced candidates."""

import statistics
import time

import torch

import solution_fast_candidate as first
import solution_ultrafast_candidate as ultra


def call(module, q, scale):
    return module.hif4_dynamic_quantize_activation(q, scale, None)


def main() -> None:
    torch.manual_seed(31)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    q = torch.randn(1024, 4096).mul_(2).round_().div_(2).clamp_(-6, 6)
    scale = torch.rand(1024, 256).mul_(1.9).add_(0.1)
    for _ in range(3):
        call(first, q, scale)
        call(ultra, q, scale)

    first_times = []
    ultra_times = []
    for index in range(20):
        order = ((first, first_times), (ultra, ultra_times))
        if index & 1:
            order = tuple(reversed(order))
        for module, samples in order:
            start = time.perf_counter()
            result = call(module, q, scale)
            samples.append((time.perf_counter() - start) * 1000.0)
            # Force a value read outside the timed region and retain no output.
            assert result["mant"].shape[-4:] == (64, 8, 2, 4)

    first_ms = statistics.median(first_times)
    ultra_ms = statistics.median(ultra_times)
    print(f"first median: {first_ms:.3f} ms")
    print(f"ultra median: {ultra_ms:.3f} ms")
    print(f"ultra improvement: {(first_ms / ultra_ms - 1.0) * 100.0:.2f}%")

    expected = call(first, q[:8], scale[:8])
    actual = call(ultra, q[:8], scale[:8])
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    print("output equivalence: OK")


if __name__ == "__main__":
    main()
