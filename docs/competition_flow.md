# 赛题流程与评分约束（2.1）

本文记录当前收到的 2.1 节赛题信息，作为实现、性能优化和提交检查的约束基线。若后续正式规则与本文冲突，以最终上线版本为准。

## 1. 上传代码

提交物必须是 `solution.zip`，其中 `solution.py` 位于压缩包根目录。平台通过以下六个公开函数调用算法：

| 阶段 | 函数 | 职责 |
| --- | --- | --- |
| Linear 校准 | `hif4_calibration_and_quantize_weight` | 使用 NVFP4 权重和校准激活数据生成 HiF4 权重参数及在线激活状态 |
| Linear 在线 | `hif4_dynamic_quantize_activation` | 使用 NVFP4 在线激活和校准状态生成 HiF4 激活参数 |
| Attention 校准 | `hif4_calibration_attention` | 使用 NVFP4 Q/K/V 校准数据生成三个在线状态 |
| Attention 在线 | `hif4_dynamic_quantize_q` | 生成 Q 的 HiF4 参数 |
| Attention 在线 | `hif4_dynamic_quantize_k` | 生成 K 的 HiF4 参数 |
| Attention 在线 | `hif4_dynamic_quantize_v` | 生成 V 的 HiF4 参数 |

可通过以下命令生成提交包：

```bash
python tools/package_solution.py
```

## 2. 判题执行

平台会对全部数据运行选手算法，并记录三个方面的结果。

### 2.0 数据组织与评分切分

平台分别提供 Linear 数据和 Attention 数据；两类数据各有 `N` 组，当前尚未给出 `N` 的具体值。

```text
Linear group
├── weight（1 份）
├── calibration set
└── test set

Attention group
├── Q
│   ├── calibration set
│   └── test set
├── K
│   ├── calibration set
│   └── test set
└── V
    ├── calibration set
    └── test set
```

校准集只用于生成 Weight HiF4 参数及后续 dynamic 函数需要的 state，**不参与 MSE 评分**；最终得分只由测试集产生。不得据此假定校准阶段没有时间成本，单次提交仍受下述总运行时间限制。

### 2.1 参数合法性

每次返回都必须满足以下离散值域，任意 case 违规都会使本次提交无效：

| 字段 | 约束 |
| --- | --- |
| `scale_factor` | HiF4 E6M2 scale |
| `scale_lv2` | `{1, 2}` |
| `scale_lv3` | `{1, 2}` |
| `sign` | `{-1, 0, 1}` |
| `mant` | `{0, 0.25, 0.5, ..., 1.75}` |

提交前应使用赛题附录提供的 `self_check.py` 校验所有输出。

### 2.2 执行时间

单次提交的**总运行时间上限为七分钟**，不对单个用例另设运行时间限制。总时间超限会判定提交失败。

该规则允许在不同组之间合理分配计算预算，但不意味着某个 case 可以无界运行：任何慢用例都会挤占全局七分钟预算，且初赛第二阶段增加用例后会放大逐 case 固定开销。因此 dynamic 路径不能使用无界搜索，校准阶段生成的固定信息应尽可能复用，并应避免不必要的反量化、中间 Tensor 和设备同步。

### 2.3 误差目标

记源格式为 NVFP4（`S`），目标格式为 HiF4（`T`）：

```text
X_FP32 = Dequant_FP32(X_S)
X_hat_T = Dequant_FP32(f_{S -> T}(X_S))
```

Linear 问题优化目标为：

```text
min MSE(A_FP32 @ B_FP32^T, A_hat_T @ B_hat_T^T)
```

Attention 问题优化目标为：

```text
MSE(
    Attn(Q_FP32, K_FP32, V_FP32),
    Attn(Q_hat_T, K_hat_T, V_hat_T),
)
```

优化目标是算子输出误差而非单个 Tensor 的重构误差。因此 calibration state 可以保存权重、head 或 channel 相关的纯数据统计量，以支持联合误差优化。

## 3. 计分

- 每个用例将选手算法的 MSE score 与标准 HiF4 量化函数比较；最终得分为各用例 score 提升百分比之和。
- 优于标准量化可获得正向提升；劣于标准量化会按劣化程度产生负分。
- 参数非法或总执行时间超限会导致提交无效或不计分。
- 排名按最终得分从高到低排列，具体计算方式以 2.2 节及最终上线规则为准。

这意味着实现必须同时守住三个门槛：**参数始终合法、运行不超时、所有 case 尽量不劣于基线**。平均误差改善不能抵消参数非法或超时风险。

## 4. 禁止方法与审核红线

**禁止以任何形式计算 `A @ W`，再利用 `A @ W` 拟合或反推出 `Q(A)`。** 该做法违背赛题目标；工作组会审核提交代码，发现后按弃赛处理。

本项目据此采用以下强制约束：

- `hif4_dynamic_quantize_activation` 只能依据当前 Activation、合法的 calibration state 以及直接从这些数据得到的统计量生成 `Q(A)`；
- 不得在显式或隐式路径中构造 Activation 与 Weight 的矩阵乘结果作为量化目标、监督信号、搜索损失或参数选择依据；
- 不得通过分块乘法、`einsum`、卷积、第三方算子、预计算查找表或其他等价变换绕过该限制；
- calibration state 即使包含固定 Weight 相关信息，也不得被用于在线还原或近似 `A @ W` 后反推 Activation 量化结果；
- 代码评审和实验记录必须能说明动态量化参数仅来自合规的局部/分组统计或预先校准的固定参数。

允许利用最终 Linear 输出 MSE 理解评分目标，但算法实现和参数搜索必须遵守上述信息边界。后续任何优化方案在落地前都应先完成一次“是否直接或间接使用 `A @ W` 拟合 `Q(A)`”的合规检查。

## 5. 性能阶段提示

在线推理速度是核心业务目标。初赛第二阶段会扩充测试用例集，决赛还会加入性能评分，因此不能把“当前用例未超时”当作最终性能标准。

实现选择遵循以下优先级：

1. dynamic 函数使用固定次数、可向量化的 Tensor 运算，不采用随输入规模增长的候选搜索；
2. 将允许离线完成的统计、阈值和策略选择放入 calibration 阶段，并通过纯数据 state 传递；
3. 避免 CPU/GPU 往返、隐式同步、逐元素 Python 循环以及重复 materialize 大型 Tensor；
4. 同时基准 calibration、单次 dynamic 和全数据总耗时，并覆盖新增 shape 与长序列场景；
5. 性能优化不得牺牲参数合法性，也不得引入上述禁止的信息路径。

性能验收以七分钟全局预算为硬门槛；开发时应保留安全余量，而不是让本地数据集耗时贴近上限。由于不存在单 case 限时，基准报告还应记录最慢 case，防止极端 shape 消耗大部分全局预算。

## 6. 平台返回

判题器计算最终得分并返回平台前端，由网页进行可视化展示。

## 尚待补充的信息

- 2.2 节的精确 score 提升百分比公式；
- E6M2 scale 的具体编码、舍入、饱和及特殊值规则；
- 标准 HiF4 基线算法或可观测基线结果；
- `self_check.py`、测试 shape、硬件环境与计时边界；
- 七分钟总运行时间的准确计时边界（是否包含导入、校准、同步与打包）以及 state 的大小/序列化限制。
