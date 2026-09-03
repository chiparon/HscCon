"""Ablate guarded three-base search by operand on synthetic/public data."""

from __future__ import annotations

import argparse
import importlib.util
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from tools import hif4_benchmark as hb


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load(ROOT / "solution.py", "policy_base")
MULTI = _load(
    ROOT / "workbench" / "solution_v2_multibase_candidate.py", "policy_multi"
)


class Policy:
    def __init__(self, roles: frozenset[str]) -> None:
        self.roles = roles

    def _q(self, role: str, quant: torch.Tensor, scale: torch.Tensor):
        if role not in self.roles and role.upper() not in self.roles:
            return BASE._quantize_hif4(quant, scale)
        quantizer = (
            MULTI._quantize_hif4_five
            if "5" in self.roles or role.upper() in self.roles
            else MULTI._quantize_hif4
        )
        return quantizer(quant, scale)

    def hif4_calibration_and_quantize_weight(self, wq, ws, calib):
        del calib
        return {
            "weight_params": self._q("w", wq, ws),
            "activation_state": {},
        }

    def hif4_dynamic_quantize_activation(self, aq, ass, state):
        del state
        return self._q("a", aq, ass)

    def hif4_calibration_attention(self, calib, qh, kvh, hd):
        del calib, qh, kvh, hd
        return {"q_state": {}, "k_state": {}, "v_state": {}}

    def hif4_dynamic_quantize_q(self, q, s, h, d, state):
        del h, d, state
        return self._q("q", q, s)

    def hif4_dynamic_quantize_k(self, q, s, h, d, state):
        del h, d, state
        return self._q("k", q, s)

    def hif4_dynamic_quantize_v(self, q, s, h, d, state):
        del h, d, state
        return self._q("v", q, s)


def synthetic(profile: str, seed: int) -> None:
    suite = hb.build_synthetic_suite(seed=seed, profile=profile)
    policies = {
        "base": frozenset(),
        "w": frozenset("w"),
        "a": frozenset("a"),
        "A": frozenset("A"),
        "wa": frozenset(("w", "a")),
        "wav": frozenset(("w", "a", "v")),
        "wav5": frozenset(("w", "a", "v", "5")),
        "Wav": frozenset(("W", "a", "v")),
        "wAv": frozenset(("w", "A", "v")),
        "q": frozenset("q"),
        "k": frozenset("k"),
        "v": frozenset("v"),
        "qk": frozenset(("q", "k")),
        "qkv": frozenset(("q", "k", "v")),
    }
    reports = {}
    for name, roles in policies.items():
        prepared = hb._prepare_solution(Policy(roles), suite)
        reports[name] = hb.evaluate_quality(prepared, suite)
    base = reports["base"]
    print(f"synthetic profile={profile} seed={seed}")
    print("policy linear_mse linear_gain attention_mse attention_gain")
    for name, report in reports.items():
        linear = report.mean_linear_mse
        attention = report.mean_attention_mse
        linear_gain = (base.mean_linear_mse - linear) / base.mean_linear_mse
        attention_gain = (
            (base.mean_attention_mse - attention) / base.mean_attention_mse
        )
        print(
            f"{name:>5} {linear:.9g} {linear_gain:+.3%} "
            f"{attention:.9g} {attention_gain:+.3%}"
        )


def _nv(pair: list[torch.Tensor]) -> hb.NVFP4Tensor:
    return hb.NVFP4Tensor(pair[0], pair[1])


def public_data() -> None:
    data_dir = ROOT / ".tmp" / "quantizer-public" / "data"
    linear = torch.load(data_dir / "linear.pt", weights_only=True, map_location="cpu")[0]
    attention = torch.load(data_dir / "attn.pt", weights_only=True, map_location="cpu")[0]
    policies = {
        "base": frozenset(),
        "w": frozenset("w"),
        "a": frozenset("a"),
        "A": frozenset("A"),
        "wa": frozenset(("w", "a")),
        "wav": frozenset(("w", "a", "v")),
        "wav5": frozenset(("w", "a", "v", "5")),
        "Wav": frozenset(("W", "a", "v")),
        "wAv": frozenset(("w", "A", "v")),
        "q": frozenset("q"),
        "k": frozenset("k"),
        "v": frozenset("v"),
        "qk": frozenset(("q", "k")),
        "qkv": frozenset(("q", "k", "v")),
    }
    results: dict[str, tuple[float, float]] = {}
    for name, roles in policies.items():
        policy = Policy(roles)
        wq, ws = linear["weight"]
        linear_calib = linear["calib_activation_list"]
        linear_result = policy.hif4_calibration_and_quantize_weight(wq, ws, linear_calib)
        weight = _nv(linear["weight"])
        linear_errors = []
        for pair in linear["test_activation_list"]:
            act = _nv(pair)
            ap = policy.hif4_dynamic_quantize_activation(
                pair[0], pair[1], linear_result["activation_state"]
            )
            linear_errors.append(
                hb.linear_operator_mse(
                    act, weight, ap, linear_result["weight_params"]
                )
            )

        qh = attention["q_num_heads"]
        kvh = attention["kv_num_heads"]
        hd = attention["head_dim"]
        states = policy.hif4_calibration_attention(
            attention["calib"], qh, kvh, hd
        )
        attn_errors = []
        for sample in attention["test"]:
            q, k, v = (_nv(sample[key]) for key in ("q", "k", "v"))
            qp = policy.hif4_dynamic_quantize_q(
                q.quant, q.scale, qh, hd, states["q_state"]
            )
            kp = policy.hif4_dynamic_quantize_k(
                k.quant, k.scale, kvh, hd, states["k_state"]
            )
            vp = policy.hif4_dynamic_quantize_v(
                v.quant, v.scale, kvh, hd, states["v_state"]
            )
            source = hb._attention(
                hb.dequantize_nvfp4(q).unsqueeze(0),
                hb.dequantize_nvfp4(k).unsqueeze(0),
                hb.dequantize_nvfp4(v).unsqueeze(0),
                qh,
                kvh,
                hd,
            )
            converted = hb._attention(
                hb.dequantize_hif4(qp, q.shape).unsqueeze(0),
                hb.dequantize_hif4(kp, k.shape).unsqueeze(0),
                hb.dequantize_hif4(vp, v.shape).unsqueeze(0),
                qh,
                kvh,
                hd,
            )
            attn_errors.append(float((source - converted).square().mean().item()))
        results[name] = (
            statistics.fmean(linear_errors), statistics.fmean(attn_errors)
        )

    base_linear, base_attention = results["base"]
    print("public mini sample")
    print("policy linear_mse linear_gain attention_mse attention_gain")
    for name, (linear_mse, attention_mse) in results.items():
        print(
            f"{name:>5} {linear_mse:.9g} "
            f"{(base_linear-linear_mse)/base_linear:+.3%} "
            f"{attention_mse:.9g} "
            f"{(base_attention-attention_mse)/base_attention:+.3%}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "default", "stress"))
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.profile:
        synthetic(args.profile, args.seed)
    if args.public:
        public_data()


if __name__ == "__main__":
    main()
