"""Ablation: accepted Linear and Attention paths, fixed-base V."""

import sys
from pathlib import Path


_REPOSITORY_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, _REPOSITORY_ROOT)

from workbench.solution_v2_fused_candidate import *  # noqa: F403
from workbench import solution_v2_attention_candidate as _attention


def hif4_dynamic_quantize_v(
    v_quant,
    v_scale,
    kv_num_heads,
    head_dim,
    v_state,
):
    return _attention.hif4_dynamic_quantize_v(
        v_quant, v_scale, kv_num_heads, head_dim, v_state
    )
