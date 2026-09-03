# v2 Linear calibration-aware candidate report

## Scope and information boundary

Candidate: `workbench/solution_v2_linear_candidate.py`.

The implementation never forms `A @ W`, a tiled equivalent, or an output
surrogate inside either public quantization function.  It uses only two
operand-local diagonal statistics:

1. calibration activation second moments for static weight conversion;
2. column energy of the already converted weight for dynamic activation
   conversion.

The dedicated evaluator computes `A @ W.T` only after both submissions have
returned their parameters, solely to measure the official operator-level MSE.

## Diagonal operator-loss approximation

For `Y = A W^T`, write the conversion errors as `Delta A = A - A_hat` and
`Delta W = W - W_hat`.  Ignoring the second-order cross term and off-diagonal
channel covariance gives

```text
E ||A DeltaW^T||_F^2 ~= sum[o,c] m[c] * DeltaW[o,c]^2
m[c] = E_calibration[A[:,c]^2]

||DeltaA What^T||_F^2 ~= sum[t,c] e[c] * DeltaA[t,c]^2
e[c] = sum[o] What[o,c]^2
```

Both objectives remain separable over each HiF4 block.  The existing exact
three-effective-scale DP is therefore retained, with every four-value residual
multiplied by its channel importance before reduction.

Raw importance is normalized to mean one inside each 64-channel block, clipped
to `[0.01, 64]`, and square-root compressed.  The square root was selected as
a shrinkage rule against finite-calibration outliers; it improved the public
mini-sample more than either full weighting or fourth-root weighting.

For static weight only, the candidate evaluates the nominal E6M2 base and its
next larger E6M2 neighbour.  Dynamic activation keeps the nominal base and only
changes the hierarchy decisions, keeping online complexity close to v1.

## Frozen state

For input width `C`, `activation_state` is:

```text
{
    "version": 3,
    "column_energy": Float32Tensor[C/64, 8, 2, 4]
}
```

The tensor is detached CPU data when the input is CPU.  For the public
`C=2048` Linear sample it contains 2048 floats (8 KiB).  It is the normalized,
clipped, square-root-compressed column energy of `W_hat`, not the source weight
and not any Linear output.

## Experiment design

`tools/benchmark_v2_linear.py` consumes the real public `linear.pt` payload.
It builds all states before quality evaluation, warms both implementations,
then shuffles and interleaves baseline/candidate calibration and dynamic calls
with seed `20260903`.  Data loading and operator MSE matmuls are excluded from
the quantizer timer.  The final run used 8 CPU threads, 2 warmups, and 3 timing
repetitions.

Public mini-sample shapes:

```text
weight: [8192, 2048]
calibration tokens: 10, 128, 512, 1024, 1024
test tokens:        10, 128, 512, 1024, 1024
```

## Results against repository v1

Final two-base candidate:

| Test | v1 operator MSE | Candidate MSE | Improvement |
| ---: | ---: | ---: | ---: |
| 0 | 0.0195779186 | 0.0148144513 | 24.331% |
| 1 | 0.0153729934 | 0.0109311296 | 28.894% |
| 2 | 0.0155305043 | 0.0109895589 | 29.239% |
| 3 | 0.0145131089 | 0.0102100614 | 29.649% |
| 4 | 0.0154408589 | 0.0109344628 | 29.185% |

Aggregate summed-MSE improvement is **28.042%**, and **5/5** cases exceed the
required 5% gate.  Median quantizer latency was:

| Stage | v1 | Candidate | Ratio |
| --- | ---: | ---: | ---: |
| Calibration | 196.138 ms | 407.973 ms | 2.080x |
| Dynamic call | 8.885 ms | 8.844 ms | 0.995x |
| One full suite | 264.536 ms | 459.725 ms | 1.738x |

The public self-check accepted the calibration result, frozen state, and all
five dynamic outputs: 0 failures in 6 checks.

## Iteration evidence and fusion recommendation

| Variant | Aggregate improvement | Cases >5% | Suite ratio |
| --- | ---: | ---: | ---: |
| Weighted hierarchy, nominal base only | 13.626% | 5/5 | 1.207x |
| Nominal + lower neighbour | 11.714% | 5/5 | 1.688x |
| Nominal + upper neighbour, fourth-root weights | 23.562% | 5/5 | 1.699x |
| Nominal + upper neighbour, square-root weights | **28.042%** | **5/5** | 1.738x |
| Four-weight-base / two-activation-base prototype | 28.865% | 5/5 | 2.899x |

The final strategy keeps the square-root weighted DP, a two-candidate static
base search, and a one-candidate online path.  If fusion with the Attention
candidate makes the total budget tight, the nominal-base-only variant is the
preferred fallback because it still clears the accuracy gate in all five real
tests while reducing the suite ratio from 1.738x to 1.207x.

## Failure points

The repository's synthetic outlier cases inject test outliers independently of
calibration.  Across three fixed seeds, the two-base candidate improved 8/9
Linear cases but exceeded 5% in only 4/9 and regressed one case by 0.43%; its
summed improvement was about 4.17%.  This is expected when a diagonal
calibration statistic is not predictive of test-time channel importance.
Therefore this direction should be fused as a public-data-qualified v2 path,
not treated as a universal no-regression guarantee.  A conservative fusion can
select the nominal-base-only path for weakly anisotropic or unstable
calibration statistics.
