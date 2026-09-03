# HiF4 v2 fusion report

## Decision

The selected standalone candidate is
`workbench/solution_v2_fused_candidate.py`.  It combines:

- Linear: calibration-derived diagonal Smooth scaling plus the same normalized
  H64 on activation and weight.  The floating operator is preserved as
  `(A / s) H @ ((W * s) H).T = A @ W.T`.
- Attention Q/K: reciprocal Smooth-QK plus the same per-64 H64.  A
  calibration-only gate enables this path for stable GQA and otherwise returns
  exactly to fixed-base conversion.
- Attention V: guarded predecessor/nominal/successor E6M2 search, only when the
  Q/K transform passes the gate.

No test tensor is available to calibration, no `A @ W.T` is evaluated inside
the quantizer, and the gate uses only Q/K second moments from calibration.

## Paired public-mini result versus current root

The official-layout comparator scores operator MSE/NMSE, not tensor SSE.

| Response | Macro gain | Worst case |
|---|---:|---:|
| Linear `A @ W.T` | +75.026% | +71.378% |
| Full GQA | +39.069% | +35.610% |
| Causal GQA | +37.736% | +36.465% |
| All 15 paired responses | +50.610% | +35.610% |

The one-pass quantizer-time ratio was 1.509x.  All parameter legality, state,
and MSE/NMSE-consistency checks passed.  Full data are in
`workbench/benchmark_v2_fused_final_official.json`.

## Multi-seed synthetic validation

Each row summarizes seeds 20260903, 20260904, and 20260905.  Timing uses the
interleaved benchmark's median suite time.  Gate rejection produces an exact
zero-gain fallback, so the worst case is zero rather than a regression.

| Profile | Median Linear macro | Median Attention macro | Median overall macro | Median runtime ratio | Worst case |
|---|---:|---:|---:|---:|---:|
| default | +31.999% | +8.981% | +20.725% | 1.526x | 0.000% |
| stress | +27.658% | +22.734% | +25.296% | 1.690x | 0.000% |

The six reports are
`workbench/benchmark_v2_fused_final_{default,stress}_{seed}.json`.

## Gate design and ratio ablation

The earlier unconditional Attention candidate regressed on independent MHA and
outlier draws because a transform that improved one calibration realization
was applied to every layout.  Checking in-sample operator output alone also did
not protect against calibration-to-test shift.

The selected gate therefore requires all of the following before storing the
transform:

1. GQA ratio at least 4;
2. at least two independent calibration entries;
3. aggregate `std(log2(smooth)) <= 0.22`;
4. mean per-channel cross-sample log-smooth standard deviation `<= 0.16`.

Ratio 8 rejects the ordinary 4:1 GQA cases.  Ratio 4 retained positive test
gains of +34.775%, +26.943%, and +5.651% over three default seeds while the
stability thresholds rejected the unstable outlier cases.  Independently, the
structured 8:1 stress GQA cases gained +41.623%, +31.270%, and +42.863%, showing
that the gate does not obtain robustness by disabling all Attention transforms.
MHA and rejected GQA use the exact fixed path; V multibase is disabled with
them as well.

## Fusion ablations

| Change | Accuracy effect | Timing effect | Decision |
|---|---:|---:|---|
| Old Linear diagonal weighting | +28.260% public macro vs root | about 1.7x suite | Replaced |
| Linear Smooth + H64, fixed base | +75.026% public macro vs root | selected lean core | Adopt |
| Add diagonal weighted DP to transformed Linear | only about 1.8% incremental error reduction | Linear suite 1.429x -> 1.744x in paired run | Reject |
| Add `+1` base to transformed W and A | about 4.7% incremental error reduction | Linear suite about 2.147x | Reject |
| Attention Smooth-QK + H64, fixed V | full +32.246%, causal +31.108% vs root | moderate | Adopt behind gate |
| Add guarded V three-base search | +10.070% full and +9.619% causal incremental error reduction | final multi-seed suite stays <=1.690x median | Adopt behind same gate |

## Verification

- Official self-check: 22/22.
- Candidate is one standalone Python file; the identical submission copy is at
  `workbench/v2_fused_submission/solution.py`.
- State contains only supported dense CPU tensors and primitive containers on
  the public CPU run; no calibration tensor storage is aliased.
- Scale-factor values remain finite legal E6M2 values and all HiF4 shapes and
  alphabets pass the official checker.
