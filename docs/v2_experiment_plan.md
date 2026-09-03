# HiF4 v2 矩阵与 Attention 适配实验计划

## 目标与采纳门槛

v1 是逐 Tensor 重构误差的精确基线。v2 将计算预算从平台实测的
144 秒提高到约 230–280 秒，换取 Linear `A @ W.T` 与 Attention
`softmax(QK.T/sqrt(d))V` 的算子级误差下降。

候选方向只有同时满足以下条件才进入融合版本：

1. 所有输出通过 HiF4 shape、E6M2、lv2/lv3、sign、mant 合法性检查；
2. 相对 v1 的 case-level operator MSE 宏平均改善严格大于 5%；
3. 多 seed 改善中位数为正，且不能靠单一 case 的极端收益掩盖普遍退化；
4. 不计算或近似 `A @ W` 来反推 activation 量化参数；
5. 本地总耗时目标不超过 v1 的 1.9 倍，除非精度收益显著高于 5%。

每个 case 的精度收益定义为：

```text
gain_percent = 100 * (MSE_v1 - MSE_candidate) / MSE_v1
```

主指标是跨 case、数据分布和随机 seed 的宏平均，而不是把不同数量的
Tensor 调用当作独立重复。计时重复只估计延迟噪声，不计入精度样本数。

## 第一轮独立方向

| 方向 | 主要因子 | 预期适配对象 |
| --- | --- | --- |
| Linear 灵敏度 | 校准 activation 二阶矩、weight 列能量、权重截断 | `A @ W.T` |
| Attention 灵敏度 | Q/K 对端二阶矩、GQA head 映射、V 独立策略 | SDPA/MHA/GQA |
| E6M2 邻域 | 单 base、3-base、5-base，固定候选 | 通用重构与算子误差 |
| 离群分布策略 | block `amax/RMS`、稀疏 outlier mask | 长尾数据 |
| 融合内核 | 候选共享 restore、loss 与 metadata | 运行时间 |
| 审核与泛化 | 多 seed、边界、shape、值域、时间交错 | 防止代理数据过拟合 |

第一轮各方向不得修改根目录 `solution.py`，只输出独立候选和可复现实验。

## 数据与阻断

精度评估至少覆盖：

- Linear：normal、宽/高矩阵、channel scale 不均衡、稀疏 outlier；
- Attention：MHA、2/4/8 倍 GQA、head_dim 64/128/256、短/长序列；
- logits：近均匀 softmax、尖峰 softmax、Q/K 尺度不平衡；
- 边界：全零、E6M2 上下界、S1P2 半舍入点和饱和点。

使用固定 seed 集合生成数据。候选与 v1 在相同 case 上配对比较；每轮计时
对候选顺序随机化并交错运行，warm-up 与数据生成不进入计时窗口。

## 融合与循环

1. 独立筛选：每个方向单独对比 v1，未超过 5% 的方向不直接进入融合；
2. 二因子融合：对通过门槛的方向做组合，检查互相增强或抵消；
3. 性能融合：共享反量化、base 候选和 DP 中间量，压缩到平台目标时间；
4. 泛化复核：使用未参与选择的 seed 和 stress shape 再测一次；
5. 发布：只有最终融合版本重新通过六接口、值域、operator MSE、性能和打包
   检查后，才覆盖根目录 `solution.py` 并标记为 v2。

平台提交不在本轮自动执行范围；生成包后等待明确的提交授权。
