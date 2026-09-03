"""Ablation facade enabling both diagonal-importance paths."""

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from workbench import solution_v2_linear_transform_candidate as _candidate

_candidate._USE_WEIGHT_IMPORTANCE = True
_candidate._USE_ACTIVATION_IMPORTANCE = True
hif4_calibration_and_quantize_weight = _candidate.hif4_calibration_and_quantize_weight
hif4_dynamic_quantize_activation = _candidate.hif4_dynamic_quantize_activation
hif4_calibration_attention = _candidate.hif4_calibration_attention
hif4_dynamic_quantize_q = _candidate.hif4_dynamic_quantize_q
hif4_dynamic_quantize_k = _candidate.hif4_dynamic_quantize_k
hif4_dynamic_quantize_v = _candidate.hif4_dynamic_quantize_v
