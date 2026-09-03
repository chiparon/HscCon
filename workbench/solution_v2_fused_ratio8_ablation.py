"""Ablation facade that raises the accepted GQA ratio from four to eight."""

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from workbench import solution_v2_fused_candidate as _fused


_fused._ATTENTION_MIN_GQA_RATIO = 8
hif4_calibration_and_quantize_weight = _fused.hif4_calibration_and_quantize_weight
hif4_dynamic_quantize_activation = _fused.hif4_dynamic_quantize_activation
hif4_calibration_attention = _fused.hif4_calibration_attention
hif4_dynamic_quantize_q = _fused.hif4_dynamic_quantize_q
hif4_dynamic_quantize_k = _fused.hif4_dynamic_quantize_k
hif4_dynamic_quantize_v = _fused.hif4_dynamic_quantize_v
