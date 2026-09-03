import importlib.util

import torch

from tools import hif4_benchmark as b


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(profile="smoke"):
    base = load("solution.py", "base")
    cand = load("workbench/solution_v2_attention_candidate.py", "cand")
    suite = b.build_synthetic_suite(profile=profile)
    for case in suite.attention:
        payload = b._attention_calibration_payload(case)
        base_state = base.hif4_calibration_attention(
            payload, case.q_num_heads, case.kv_num_heads, case.head_dim
        )
        state = cand.hif4_calibration_attention(
            payload, case.q_num_heads, case.kv_num_heads, case.head_dim
        )
        print(case.name)
        for index, (q, k, v) in enumerate(case.test):
            rows = []
            for module, states in ((base, base_state), (cand, state)):
                qp = module.hif4_dynamic_quantize_q(
                    q.quant, q.scale, case.q_num_heads, case.head_dim, states["q_state"]
                )
                kp = module.hif4_dynamic_quantize_k(
                    k.quant, k.scale, case.kv_num_heads, case.head_dim, states["k_state"]
                )
                vp = module.hif4_dynamic_quantize_v(
                    v.quant, v.scale, case.kv_num_heads, case.head_dim, states["v_state"]
                )
                q0, k0, _ = map(b.dequantize_nvfp4, (q, k, v))
                q1 = b.dequantize_hif4(qp, q.shape)
                k1 = b.dequantize_hif4(kp, k.shape)
                hq = q0.reshape(-1, case.q_num_heads, case.head_dim).transpose(0, 1)
                hk = k0.reshape(-1, case.kv_num_heads, case.head_dim).transpose(0, 1)
                hqh = q1.reshape(-1, case.q_num_heads, case.head_dim).transpose(0, 1)
                hkh = k1.reshape(-1, case.kv_num_heads, case.head_dim).transpose(0, 1)
                if case.q_num_heads != case.kv_num_heads:
                    repeat = case.q_num_heads // case.kv_num_heads
                    hk = hk.repeat_interleave(repeat, 0)
                    hkh = hkh.repeat_interleave(repeat, 0)
                rows.append(
                    (
                        torch.mean((q0 - q1) ** 2).item(),
                        torch.mean((k0 - k1) ** 2).item(),
                        torch.mean(
                            ((hq @ hk.transpose(-1, -2)) - (hqh @ hkh.transpose(-1, -2))) ** 2
                        ).item(),
                        b.attention_operator_mse(
                            (q, k, v),
                            (qp, kp, vp),
                            case.q_num_heads,
                            case.kv_num_heads,
                            case.head_dim,
                        ),
                    )
                )
            print(index, rows)


run()
