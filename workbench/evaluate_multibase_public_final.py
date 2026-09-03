"""Per-test public mini-sample audit for the selected multibase candidate."""

from __future__ import annotations

import importlib.util
import statistics
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import hif4_benchmark as hb


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def nv(pair) -> hb.NVFP4Tensor:
    return hb.NVFP4Tensor(pair[0], pair[1])


def run(module, linear, attention, score: bool):
    wq, ws = linear["weight"]
    lr = module.hif4_calibration_and_quantize_weight(
        wq, ws, linear["calib_activation_list"]
    )
    linear_errors = []
    for pair in linear["test_activation_list"]:
        ap = module.hif4_dynamic_quantize_activation(
            pair[0], pair[1], lr["activation_state"]
        )
        if score:
            linear_errors.append(
                hb.linear_operator_mse(
                    nv(pair), nv(linear["weight"]), ap, lr["weight_params"]
                )
            )

    qh, kvh, hd = (
        attention["q_num_heads"], attention["kv_num_heads"], attention["head_dim"]
    )
    states = module.hif4_calibration_attention(attention["calib"], qh, kvh, hd)
    attention_errors = []
    for sample in attention["test"]:
        items = [nv(sample[key]) for key in ("q", "k", "v")]
        params = [
            module.hif4_dynamic_quantize_q(
                items[0].quant, items[0].scale, qh, hd, states["q_state"]
            ),
            module.hif4_dynamic_quantize_k(
                items[1].quant, items[1].scale, kvh, hd, states["k_state"]
            ),
            module.hif4_dynamic_quantize_v(
                items[2].quant, items[2].scale, kvh, hd, states["v_state"]
            ),
        ]
        if score:
            source = hb._attention(
                *(hb.dequantize_nvfp4(x).unsqueeze(0) for x in items), qh, kvh, hd
            )
            converted = hb._attention(
                *(
                    hb.dequantize_hif4(p, x.shape).unsqueeze(0)
                    for p, x in zip(params, items)
                ),
                qh,
                kvh,
                hd,
            )
            attention_errors.append(float((source - converted).square().mean().item()))
    return linear_errors, attention_errors


def main() -> None:
    torch.set_num_threads(8)
    data_dir = ROOT / ".tmp" / "quantizer-public" / "data"
    linear = torch.load(data_dir / "linear.pt", weights_only=True)[0]
    attention = torch.load(data_dir / "attn.pt", weights_only=True)[0]
    baseline = load(ROOT / "solution.py", "public_final_baseline")
    candidate = load(
        ROOT / "workbench" / "solution_v2_multibase_candidate.py",
        "public_final_candidate",
    )
    base_linear, base_attention = run(baseline, linear, attention, True)
    cand_linear, cand_attention = run(candidate, linear, attention, True)
    for role, before, after in (
        ("Linear", base_linear, cand_linear),
        ("Attention", base_attention, cand_attention),
    ):
        for index, (left, right) in enumerate(zip(before, after)):
            print(
                f"{role}[{index}] {left:.10g} -> {right:.10g}; "
                f"gain={(left-right)/left:+.3%}"
            )
        left, right = statistics.fmean(before), statistics.fmean(after)
        print(f"{role} mean {left:.10g} -> {right:.10g}; gain={(left-right)/left:+.3%}")

    timings = {"baseline": [], "candidate": []}
    for repeat in range(5):
        order = (("baseline", baseline), ("candidate", candidate))
        if repeat % 2:
            order = tuple(reversed(order))
        for name, module in order:
            start = time.perf_counter()
            run(module, linear, attention, False)
            timings[name].append(time.perf_counter() - start)
    btime = statistics.median(timings["baseline"])
    ctime = statistics.median(timings["candidate"])
    print(
        f"Full mini workflow median: baseline={btime:.6f}s "
        f"candidate={ctime:.6f}s multiplier={ctime/btime:.3f}x"
    )


if __name__ == "__main__":
    main()
