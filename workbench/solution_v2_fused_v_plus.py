"""Ablation facade: fused candidate with nominal/upper-neighbour V."""

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from workbench.solution_v2_fused_candidate import *  # noqa: F403
from workbench import solution_v2_multibase_candidate as _multibase


def hif4_dynamic_quantize_v(v_quant, v_scale, kv_num_heads, head_dim, v_state):
    del kv_num_heads, head_dim, v_state
    return _multibase._quantize_hif4_search(v_quant, v_scale, (1,))
