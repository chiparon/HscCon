# HscCon

HscCon 用于参加 NVFP4 → HiF4 量化比赛。项目的首要目标是在满足输出参数合法性和运行时间限制的前提下，降低 Linear 与 Attention 的最终输出 MSE，相对标准 HiF4 量化函数取得尽可能高的 score 提升。

## 优化目标

比赛不是要求逐值复现某个唯一的 NVFP4 → HiF4 转换结果，而是允许选手自行设计 HiF4 量化算法，并按模型算子输出误差评分：

- Linear：比较 `A_FP32 @ B_FP32.T` 与 `A_hat_HiF4 @ B_hat_HiF4.T` 的 MSE；
- Attention：比较 FP32 Q/K/V 与量化后 Q/K/V 所产生 Attention 输出的 MSE；
- 每个测试 case 的结果与标准 HiF4 量化函数比较，score 提升百分比之和构成最终得分；
- 校准集用于产生权重量化参数和在线量化 state，不直接参与评分；
- 测试集用于最终评分，任一 case 劣于标准方法都可能产生负分。

因此，本项目关注的是**量化策略带来的端到端精度提升**。运行性能是必须满足的约束，也是后续阶段的评分因素，但不是用来替代精度目标的项目主线。

## 硬性约束

### 输出合法性

每个 HiF4 量化结果至少包含以下字段：

| 字段 | 合法值 |
| --- | --- |
| `scale_factor` | HiF4 E6M2 scale |
| `scale_lv2` | `1` 或 `2` |
| `scale_lv3` | `1` 或 `2` |
| `sign` | `-1`、`0` 或 `1` |
| `mant` | `0` 至 `1.75`，步长 `0.25` |

对应反量化关系为：

```text
x_hat = sign * mant * scale_lv3 * scale_lv2 * scale_factor
```

任意 case 输出非法都会导致本次提交无效。

### 禁止方法

严禁以任何形式计算或变相计算 `A @ W`，并利用其拟合、搜索或反推出 `Q(A)`。工作组会审核代码，发现该行为将按弃赛处理。

### 运行时间

单次提交的全部用例共享七分钟总运行时间，单个用例没有独立限时。初赛第二阶段会增加测试用例，决赛阶段会增加性能评分，因此算法需要兼顾精度与在线推理效率。

## 提交接口

根目录的 `solution.py` 必须实现六个公开函数：

| 场景 | 函数 |
| --- | --- |
| Linear calibration 与 Weight 量化 | `hif4_calibration_and_quantize_weight` |
| Linear Activation 在线量化 | `hif4_dynamic_quantize_activation` |
| Attention calibration | `hif4_calibration_attention` |
| Query 在线量化 | `hif4_dynamic_quantize_q` |
| Key 在线量化 | `hif4_dynamic_quantize_k` |
| Value 在线量化 | `hif4_dynamic_quantize_v` |

输入 Weight、Activation、Q、K、V 均由 NVFP4 carrier 和每 16 个元素一组的 block scale 提供。HiF4 输出每 64 个元素一组，具体 Tensor shape 与 state 数据契约见 `solution.py`。

## 数据组织

- Linear 共 `N` 组，每组包含一份 Weight、校准集和测试集；
- Attention 共 `N` 组，每组包含 Q、K、V，三者分别包含校准集和测试集；
- `N` 的具体值尚未提供。

## 打包提交

提交物是 `solution.zip`，并且 `solution.py` 必须位于压缩包根目录。使用仓库工具生成：

```bash
python tools/package_solution.py
```

完整的已知赛题规则整理在 [`docs/competition_flow.md`](docs/competition_flow.md)。

## 当前状态

仓库目前只完成接口、NVFP4 反量化辅助函数、提交打包和基础契约测试。六个比赛量化函数尚未实现；后续将根据分批提供的赛题资料逐步建立算法与评测代码。
