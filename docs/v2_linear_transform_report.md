# v2 Linear 矩阵变换候选报告

## 结论

建议融合 **RMS-Smooth + block Hadamard64** 方向。所有收益均以当前根目录
`solution.py` 为 control；与线上提交、S8 得分或第三方实现无关。

最终 standalone 候选为
`workbench/solution_v2_linear_transform_candidate.py`，SHA256：
`98C85F136FF49C4F01A085B0811B7FD33F5F597AB183E22DB8CE20B28C6D1C23`。

## 方法

令 `s` 为按输入通道得到的正数 Smooth scale，`H` 为每 64 通道独立应用的
归一化 Walsh-Hadamard 矩阵。候选在量化前计算：

```text
A' = (A / s) H
W' = (W * s) H
```

因为 `H H.T = I`，所以浮点参考下 `A' W'.T = A W.T`。随机矩阵测试的最大
乘积绝对差为 `1.24e-5`，两次 H64 的最大还原绝对差为 `2.54e-7`。

最终 scale 使用 calibration activation 与 weight 的 channel RMS，指数
`alpha=0.65`；每个 64-channel block 内先取几何中心，再将相对 log-scale
收缩到原来的 `0.5`，最后限幅到 `[1/16, 16]`。这一步避免少量 calibration
离群点把单个通道的 scale 放大后迁移到 test。

量化器没有计算 `A @ W.T`、`matmul`、`mm` 或 `einsum`。默认保留根版本的
单 base 精确 hierarchy DP；calibration state 只有标量版本号和一个 2048
元素的 Smooth tensor。

## Public mini sample（修正版 comparator）

命令：

```powershell
python tools/hif4_official_sample_benchmark.py `
  --control solution.py `
  --candidate workbench/solution_v2_linear_transform_candidate.py `
  --dataset-dir .tmp/quantizer-public/data `
  --threads 8 --warmups 1 --repeats 3 --seed 20260903 `
  --json workbench/audit_v2_linear_transform.json
```

| Linear test | v1 MSE | 候选 MSE | 收益 |
| ---: | ---: | ---: | ---: |
| 0 | 0.0195779 | 0.00450549 | +76.987% |
| 1 | 0.0153730 | 0.00440010 | +71.378% |
| 2 | 0.0155305 | 0.00384500 | +75.242% |
| 3 | 0.0145131 | 0.00369132 | +74.566% |
| 4 | 0.0154409 | 0.00355784 | +76.958% |

- Linear case 宏平均：`+75.026%`；最差 Linear case：`+71.378%`；`5/5`
  case 严格超过 5%。
- Attention 路径未修改，10 个 full/causal 指标均为 `0.000%`；因此把未触及
  的 Attention 也计入 15-case overall 后为 `+25.009%`，overall 最差为 0。
- MSE 与 NMSE 两种 gain 算法的最大差为 `1.32e-6` 个百分点。
- 修正版 comparator 的本次中位计时：Linear calibration
  `206.06 -> 291.77 ms`（`1.416x`），dynamic 总计
  `39.30 -> 60.01 ms`（`1.527x`），Linear 总计 `1.434x`；包含未改
  Attention 的全 suite 为 `1.387x`。CPU 短测有噪声，不直接外推线上秒数。
- 输出合法性与 state gate 全部通过，原始记录：
  `workbench/audit_v2_linear_transform.json`。

## 消融与失败版本

在 public 5 个 Linear case 上，固定单 base、关闭 weighting 的早期消融为：

| 变换 | case gain 宏平均 | 最差 case |
| --- | ---: | ---: |
| absmax-Smooth only | +39.248% | +35.751% |
| Hadamard64 only | +44.746% | +40.257% |
| absmax-Smooth + Hadamard64 | +74.376% | +71.016% |
| 最终 RMS-Smooth-shrink + Hadamard64 | **+75.026%** | **+71.378%** |

完整 weighted-DP + activation weighting + 上邻 E6M2 base 可将宏平均推到
`+76.393%`，但相比最终纯变换只增加 1.37 个根基准百分点，并显著增加
calibration 搜索时间；本候选不默认采纳这部分，交由融合版在总时间预算下
单独决定。

最初的未收缩 absmax 版本虽然 public 宏平均约 `+75.5%`，却在开发用随机
outlier seed 20260911 上得到 `+35.0%/-97.8%/-40.8%`，宏平均 `-34.5%`。
改为 RMS 并做 block 内 0.5 收缩后，同一开发 seed 变为
`+36.4%/+46.9%/+32.4%`。最终 hold-out 使用另外三组从未参与选择的 seed。

## 未参与选择的 multi-seed hold-out

seed 为 `112358/424242/867530`。下表给出每个 seed 的 Linear case 宏平均；
对应 JSON 为 `workbench/v2_linear_transform_holdout_{profile}_seed*.json`。

| profile | seed 112358 | seed 424242 | seed 867530 | pooled 宏平均 | pooled 最差 | >5% cases | suite 时间比中位数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| default | +31.887% | +30.243% | +34.192% | **+32.107%** | +25.986% | 9/9 | 1.373x |
| stress | +27.015% | +26.820% | +26.719% | **+26.851%** | +15.033% | 12/12 | 1.375x |

21 个 hold-out Linear case 全部超过 5%，seed-level 宏平均中位数分别为
`+31.887%` 和 `+26.820%`，没有观察到回退。

## 合法性

候选以自身文件直接注入官方 `self_check.py`，最终结果为 `22/22`：Linear
`6/6`、Attention `16/16`。候选只依赖 Python 标准库与 PyTorch，是单文件
standalone 实现，不导入根 `solution.py` 或其他 workbench 文件。
