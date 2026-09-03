"""Structured Attention data for MHA/GQA and head dimensions 64/128/256."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tools.hif4_benchmark import quantize_synthetic_nvfp4
from workbench.benchmark_v2_attention import _load, _score


def _pair(values: torch.Tensor) -> list[torch.Tensor]:
    result = quantize_synthetic_nvfp4(values)
    return [result.quant, result.scale]


def _sample(
    generator: torch.Generator,
    tokens: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> dict[str, list[torch.Tensor]]:
    repeats = q_heads // kv_heads
    coordinate = torch.linspace(-1.0, 1.0, head_dim)
    phase = torch.arange(kv_heads, dtype=torch.float32).unsqueeze(-1) * 0.37
    q_gain_kv = torch.exp(1.15 * torch.sin(3.1 * coordinate + phase))
    k_gain = torch.exp(1.10 * torch.cos(2.7 * coordinate - phase))
    q_gain = q_gain_kv.repeat_interleave(repeats, dim=0)
    q_gain *= torch.exp(
        0.12 * torch.randn((q_heads, 1), generator=generator)
    )
    v_gain = torch.exp(0.55 * torch.sin(4.3 * coordinate + phase))

    # A shared low-rank token signal creates realistic Q/K correlations while
    # independent noise prevents the benchmark from rewarding a fixed code.
    rank = min(8, head_dim)
    latent = torch.randn((tokens, rank), generator=generator)
    q_projection = torch.randn((q_heads, rank, head_dim), generator=generator)
    k_projection = torch.randn((kv_heads, rank, head_dim), generator=generator)
    q = torch.einsum("tr,hrd->thd", latent, q_projection) / rank**0.5
    k = torch.einsum("tr,hrd->thd", latent, k_projection) / rank**0.5
    q += 0.35 * torch.randn((tokens, q_heads, head_dim), generator=generator)
    k += 0.35 * torch.randn((tokens, kv_heads, head_dim), generator=generator)
    v = torch.randn((tokens, kv_heads, head_dim), generator=generator)
    q = (q * q_gain).reshape(tokens, q_heads * head_dim)
    k = (k * k_gain).reshape(tokens, kv_heads * head_dim)
    v = (v * v_gain).reshape(tokens, kv_heads * head_dim)
    return {"q": _pair(q), "k": _pair(k), "v": _pair(v)}


def build_suite(seed: int = 260903):
    generator = torch.Generator().manual_seed(seed)
    specs = (
        ("mha_d64", 4, 4, 64, 32),
        ("gqa4_d128", 8, 2, 128, 48),
        ("gqa8_d256", 16, 2, 256, 64),
    )
    groups = []
    for name, q_heads, kv_heads, head_dim, tokens in specs:
        groups.append(
            {
                "key": name,
                "q_num_heads": q_heads,
                "kv_num_heads": kv_heads,
                "head_dim": head_dim,
                "calib": [
                    _sample(generator, tokens, q_heads, kv_heads, head_dim)
                    for _ in range(2)
                ],
                "test": [
                    _sample(generator, tokens + 8 * index, q_heads, kv_heads, head_dim)
                    for index in range(2)
                ],
            }
        )
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json", default="workbench/v2_attention_synthetic.json"
    )
    args = parser.parse_args()
    torch.set_num_threads(8)
    baseline_module = _load("solution.py", "synthetic_v1")
    candidate_module = _load(
        "workbench/solution_v2_attention_candidate.py", "synthetic_v2"
    )
    rows = []
    for group in build_suite():
        baseline = _score(baseline_module, group)
        candidate = _score(candidate_module, group)
        row = {"case": group["key"], "baseline": baseline, "candidate": candidate}
        for mode in ("full", "causal"):
            old = baseline[f"mean_{mode}_mse"]
            new = candidate[f"mean_{mode}_mse"]
            row[f"{mode}_improvement_percent"] = 100.0 * (old - new) / old
        rows.append(row)
    encoded = json.dumps(rows, indent=2)
    Path(args.json).write_text(encoded, encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
