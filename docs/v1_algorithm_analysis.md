# NVFP4 → HiF4 v1 算法分析

## 1. 结论先行

首版建议采用「**固定 E6M2 base + 64→8→4 层级精确 DP**」，不要照搬白皮书中基于局部 `amax` 的贪心 micro-exponent 生成。它对给定 base 和可分解的对角加权 MSE 是精确最优的，仍只需固定次数、可向量化的 Tensor 运算。

按运行路径建议：

- Weight 校准：时延不像 dynamic 路径敏感，可用 E6M2 邻域 3–5 候选 + 加权 DP。
- Linear Activation 在线：先用单 base 加权 DP；若总时间有余量，再升到 3 个 base 候选。
- Attention Q/K：用对端 K/Q 的 head-dimension 二阶矩作为对角灵敏度，做单 base 加权 DP。
- Attention V：用单 base 非加权 DP；V 最后一维的特征灵敏度在一阶近似下基本相同。

用独立的纯 Python 合成实验对比了「白皮书 `amax` 贪心」与 DP（每种分布 5,000 个 64-元素 block）：

| 分布 | 固定 base DP | 3-base DP | 5-base DP |
| --- | ---: | ---: | ---: |
| Normal | MSE 下降 12.98% | 18.64% | 20.57% |
| Student-t(3) | 10.00% | 14.50% | 15.50% |
| Normal + 1% 的 20× outlier | 2.29% | 5.74% | 6.00% |

这些数字只是算法筛选证据，不是隐藏比赛集成绩；它们说明 DP 和小邻域 base 搜索值得进入真实算子级 benchmark。

## 2. HiF4 可表示值与 E6M2

官方 HiF4 白皮书定义每 64 个元素共享一套层级 scale：

\[
\hat{x}_i=b\cdot u_{\lfloor i/8\rfloor}\cdot
v_{\lfloor i/4\rfloor}\cdot s_i\cdot m_i,
\]

其中：

- \(b\) 是无符号 E6M2 `scale_factor`；
- \(u_j\in\{1,2\}\) 是 8-way `scale_lv2`；
- \(v_k\in\{1,2\}\) 是 16-way `scale_lv3`；
- \(s_i\in\{-1,0,1\}\)；
- \(m_i\in\{0,0.25,0.5,\ldots,1.75\}\)。

E6M2 是 6-bit exponent + 2-bit mantissa，exponent bias 为 48，只有 normal，没有零和无穷：

\[
b=2^{e-48}(1+m/4),\quad e\in[0,63],\ m\in[0,3].
\]

`0b11111111` 保留为 NaN，因此有限最大值是 `0b11111110` = \(2^{15}\times1.5\)，最小值是 \(2^{-48}\)。有限正数 code `0..254` 按数值单调增加，可以安全用 code 的 `±K` 构造固定邻域，但必须 clamp 到 `[0,254]`，不能进入 NaN code。

建议的 E6M2 转换步骤是：

1. 将正数 scale 限制到 E6M2 有限范围。
2. 用 `floor(log2(x))` 求 exponent，将 significand 量化到 `{1,1.25,1.5,1.75}`。
3. mantissa 四舍五入溢出时进位到下一 exponent。
4. 最高 exponent 将 NaN code 排除，饱和到 \(2^{15}\times1.5\)。

首版可直接返回上述可表示的浮点数值，不需返回原始 8-bit code。但最终仍要以赛方 `self_check.py` 为准，因为当前仓库未包含它。

全零 block 是特殊边界：E6M2 没有零，所以应输出最小有限 base \(2^{-48}\)，并让所有 mantissa 和 sign 为 0。

格式定义来源：[HiFloat4 Format for Language Model Inference](https://hifloat.gccorg.com/docs/en/hifloat4/white_paper/hifloat4_format_for_language_model_inference.html)。

## 3. NVFP4 反量化

输入 carrier 的最后一维每 16 元素共享一个 NVFP4 scale。先在 FP32 中还原：

\[
x_{\ldots,16g+r}=q_{\ldots,16g+r}\,s_{\ldots,g}.
\]

当前 `dequantize_nvfp4` 最后强到 BF16，对量化决策不利：v1 内部搜索应用 FP32 乘法和累加，只在需要对齐赛方定义时考虑 BF16 舍入。由于赛题的 reference 先明确写了 `Dequant_FP32`，更合理的主路径是 FP32 restore。

不建议先生成完整 FP32 副本再 reshape 多次。高性能实现可以将 carrier reshape 为 `(..., C/64, 4, 16)`，与 4 个 NVFP4 scale 广播相乘，然后 view 为 `(..., C/64, 8, 2, 4)`。

## 4. 给定 base 时，64→8→4 的精确最优选择

### 4.1 元素量化

对固定有效 scale \(a=buv\)，元素的最优 S1P2 量化是：

\[
c_i=\operatorname{clip}(\operatorname{round}(4|x_i|/a),0,7),\quad
m_i=c_i/4.
\]

白皮书允许 round-half-to-even 或 round-half-away-from-zero。PyTorch `round` 是 half-to-even，适合做无分支向量化实现。随机连续输入的 tie 概率很低，但 NVFP4 输入是离散的，tie 并非不可能；提交前必须与自检器做边界对齐。

### 4.2 4-元素局部误差

有效 micro-scale 乘积只可能是 `r in {1,2,4}`。对第 \(k\) 个 4-元素小组和非负权重 \(\omega_i\)，预计算：

\[
E_{k,r}=\sum_{i\in k}\omega_i
\left(x_i-Q(x_i;br)\right)^2,
\quad r\in\{1,2,4\}.
\]

非加权 tensor reconstruction 令 \(\omega_i=1\)。Linear/Attention 的对角灵敏度近似则使用第 7–8 节定义的权重。

### 4.3 8-元素节点的两种情形

每个 lv2 节点 \(j\) 管理两个 4-元素子组 \(k=2j,2j+1\)。枚举 \(u\in\{1,2\}\)：

\[
C_{j,u}=\sum_{k\in\{2j,2j+1\}}
\min_{v\in\{1,2\}}E_{k,uv}.
\]

然后：

\[
u_j^*=\arg\min_{u\in\{1,2\}}C_{j,u},\qquad
v_k^*=\arg\min_{v\in\{1,2\}}E_{k,u_j^*v}.
\]

这不是近似：在 base 固定、loss 能按元素求和的前提下，它枚举了所有合法的 lv2/lv3 组合，因此对整个 64-元素 block 精确最优。同一 8-元素节点不能出现有效乘积 `(1,4)`，DP 正确地保留了这个结构约束。

对比之下，基于 `amax` 的贪心规则大致是：

- `lv2=2` iff 该 8-元素块 `amax > 3.5*b`；
- `lv3=2` iff 该 4-元素块 `amax > 1.75*b*lv2`。

它的目标是尽可能不饱和，却不保证 MSE 最小。少量 outlier 往往会让整个小组选择过大的 scale，这正是 DP 能获得明显收益的原因。

## 5. E6M2 base 的固定复杂度搜索

白皮书基准 base 为：

\[
b_0=Q_{E6M2}\left(\max_i|x_i|/7\right).
\]

`7 = 1.75 * 2 * 2` 是层级结构的最大局部倍率。由于 E6M2 可能向下舍入，极值仍可能饱和；元素 quantizer 必须始终 clip 到 code 7。

可将 base 搜索限定为固定 code 邻域：

\[
\mathcal B_K=\{\operatorname{E6M2Code}^{-1}
(\operatorname{clip}(code(b_0)+\delta,0,254)):
\delta\in[-K,K]\}.
\]

对每个 base 执行第 4 节 DP，选择 64-元素 block 总 loss 最小者。`K=1` 是 3-base，`K=2` 是 5-base。候选数与输入数值、shape 和分布无关，所以仍属于固定复杂度，不是无界搜索。

要特别注意：「校准集学一个全局 base code 偏移」不能替代逐 block 邻域搜索。在合成测试中，最优偏移随 block 改变，固定 `delta=0` 的平均误差反而低于任一非零全局偏移。

## 6. 三个可落地候选

### 候选 A：官方 `amax` 贪心基线

- base：`E6M2(amax/7)`。
- lv2/lv3：用第 4.3 节的 max-threshold 规则。
- mantissa：最近 S1P2 + 饱和。
- 时间：`O(N)`，一次层级 max 归约 + 一次元素量化。
- 作用：必要的正确性与性能对照，不建议作为最终 v1。

### 候选 B：单 base 精确 DP（推荐首版）

- base：`E6M2(amax/7)`。
- lv2/lv3：第 4 节的精确 DP。
- 时间：`O(N)`，对有效乘积 1/2/4 各算一次 mantissa/error，约 3 个元素候选 pass。
- 空间：最快的向量化版可保留 3 组 mantissa code，为 `O(3N)` 临时量；显存紧张时可分块或在选完 metadata 后重算一次 mantissa，换取 `O(N)` 空间。
- 优点：速度/误差性价比最好；对给定 base 和对角加权 loss 有单调不劣于贪心的数学保证。

### 候选 C：3/5-base 邻域 + 精确 DP（后续 >5% 迭代）

- base：枚举 `b0` 的 `±1` 或 `±2` E6M2 code。
- lv2/lv3：每个 base 内部做精确 DP。
- 时间：`O(3N)` 或 `O(5N)` 个 DP，等价于约 9 或 15 个元素候选 pass。
- 用法：Weight 校准优先使用 5-base；dynamic 路径先对 3-base 做真实 shape 时延测试，只在算子 MSE 比候选 B 改善超过 5% 且总时间有余量时采纳。

## 7. Linear 的校准状态与算子级 MSE

定义 weight 误差 \(E_W=W-\hat W\)、activation 误差 \(E_A=A-\hat A\)。线性层误差可写为：

\[
AW^T-\hat A\hat W^T=A E_W^T+E_A\hat W^T.
\]

精确目标包含 channel 间协方差和两项的交叉项，但完整优化会破坏 4-元素/8-元素可分解性，也容易触碰 `A @ W` 审核红线。v1 用稳定的对角二阶近似。

### 7.1 Weight 的加权量化

从校准 activation 只计算每个 input channel 的二阶矩：

\[
h_c=\mathbb E_{A_{cal}}[A_c^2].
\]

对 weight 元素 \(W_{o,c}\) 使用 \(\omega_{o,c}=h_c\)，在每个 64-元素 input-channel block 内执行加权 DP。这对应：

\[
\mathbb E\|A(W-\hat W)^T\|^2
\approx\sum_{o,c}h_c(W_{o,c}-\hat W_{o,c})^2.
\]

为避免个别 channel 的校准 outlier 导致隐藏集泛化变差，建议每 64-channel block 将 \(h\) 归一化到均值 1，然后 clip 到如 `[0.25, 4]`。这个 clip 范围是需在自建数据上测的超参，不应在首次提交中无界搜索。

### 7.2 Activation state

Weight 量化完成后，从 \(\hat W\) 计算每个 input channel 的列能量：

\[
g_c=\sum_o\hat W_{o,c}^2.
\]

在线 activation 量化对每个 \(A_{n,c}\) 使用 \(\omega_{n,c}=g_c\)。这是 \(\|E_A\hat W^T\|^2\) 的对角近似，而且使用已量化 \(\hat W\) 而不是原始 \(W\)，与上面的精确误差分解一致。

`activation_state` 只需保存：

- 归一化/截断后的非负 `channel_weight` \(g\)；
- 算法版本和固定的 base 候选半径；
- 如果真实 benchmark 证明有用，可加每 64-channel block 的有界策略标记。

不应保存原始 W、带符号的 weight row、输出投影或任何可以在线重构 `A @ W` 的数据。

### 7.3 可选校准信息

校准 activation 还可以安全地用于：

- 每 channel/group 的 RMS、`amax/RMS` 和有界分位数；
- 选择 weight clip 上限或灵敏度 tempering 强度；
- 在不查看任何 `A @ W` 结果的前提下，最小化 activation 的局部加权 reconstruction loss。

但由于单 base DP 已对每个当前 block 直接求最优 micro-scale，首版不需要依赖复杂的 calibration-fitted threshold。

## 8. Attention Q/K/V 策略

设 Q head \(h\) 映射到 KV head \(r(h)\)。GQA 通常为按顺序均匀分组：

\[
r(h)=\left\lfloor h\,H_{kv}/H_q\right\rfloor,
\]

但必须以赛方张量 layout 为准，当前仓库没有公布 Q/K/V 的具体 shape 和 head 轴位置。

### 8.1 Q 和 K

logit 的一阶扰动为：

\[
\delta L\approx(\delta Q)K^T+Q(\delta K)^T.
\]

不显式计算 `Q @ K^T`，只从 calibration 收集对端的 head-dimension 二阶矩：

\[
g^Q_{h,d}=\mathbb E[K_{r(h),d}^2],
\qquad
g^K_{r,d}=\sum_{h:r(h)=r}\mathbb E[Q_{h,d}^2].
\]

Q 的 dynamic DP 用 \(g^Q\) 加权，K 的 dynamic DP 用 \(g^K\) 加权。这会优先保留对 logit dot-product 更敏感的维度，又不需要任何 token×token 矩阵或变长搜索。

进一步可乘上每个 KV head 的 \(\mathbb E\|V_h\|^2\) 作为粗略的 softmax-output 敏感度，但这是 head 标量；当 64-元素 block 完全落在同一 head 内时，它对 DP argmin 没有影响。因此不必在 v1 中增加它。

### 8.2 V

当 attention probability \(P\) 固定时，V 误差是 \(P\delta V\)。P 对 V 的每个 feature dimension 施加相同的 token 混合，因此对最后一维量化来说，首版使用非加权 DP 即可。

若后续要优化 V，可在校准期生成「每 head/token-position 类型」的平均权重，但需要解决长度泛化和 state 大小问题，不适合首版。

### 8.3 Softmax 风险

Q/K 的 tensor MSE 下降不保证 attention output MSE 一定同比下降，因为 softmax 会改变误差传播。所以 Attention 候选只能在自建的完整 attention score 上采纳，不能只看 Q/K/V 重构 MSE。建议同时覆盖：

- 均匀 attention（小 logits）；
- 高峰 attention（大 logits）；
- MHA 与 GQA；
- head_dim 为 64/128/256；
- 短序列与长序列；
- Q/K 尺度不平衡和局部 outlier。

## 9. `A @ W` 合规边界

这个方案的合规边界应在代码中保持很清楚。

可以做：

- 对 activation 单独做 `square / abs / max / sum / mean / percentile`；
- 对 weight 单独做列平方和 \(\sum_o W_{o,c}^2\)；
- 将两者压缩成的非负、无符号、逐 channel 灵敏度用于局部 reconstruction loss；
- 用 calibration activation 的二阶矩加权 weight quantization。

不可以做：

- 计算 `A @ W.T` 或其分块/einsum/conv/查表等价形式；
- 用真实或近似的线性层 output 选择 activation 量化参数；
- 在 `activation_state` 中保存足以在线重构 weight rows 或 output projection 的数据；
- 让 dynamic 候选 loss 含任何 output-channel 维度的带符号累加。

\(g_c=\sum_o\hat W_{o,c}^2\) 只是每 channel 一个非负灵敏度，不包含 weight 符号、row 或 output 信息，不能用来恢复 `A @ W`。它满足当前赛题文档中「state 可保存 weight/channel 相关纯数据统计量」的边界。

## 10. 实现和性能注意事项

1. 所有搜索维度固定为 3 个 effective scale，以及可选的 1/3/5 个 base；不使用根据数值收敛的 Python loop。
2. 用 `abs_x * (4 / b)` 只做一次 base 归一化；`r=1,2,4` 的 mantissa code 分别是对归一化值乘 `1, 0.5, 0.25` 后 round/clip。
3. 不要在最内层使用 Python 的 64/8/4 循环；统一 view 为 `(..., blocks64, 8, 2, 4)` 后在 Tensor 维度上 reduce。
4. 避免 `.item()`、Python bool 和 CPU/GPU 往返。E6M2 生成、候选选择和 metadata gather 全部在输入 device 上完成。
5. `scale_factor/scale_lv2/scale_lv3/mant/sign` 直接用 reshape 得到规定 shape，避免 permute 后的隐式 contiguous copy。
6. 对齐实际 dtype：误差累加用 FP32，metadata 的输出 dtype 则服从 `self_check.py`。
7. state 中的 channel/head 权重应预先移到对应 device/dtype；如果赛方会序列化 state，则要测试 tensor 是否被允许，否则使用纯容器 + 数值数组。

## 11. 采纳门槛与迭代顺序

建议使用以下严格顺序：

1. 实现候选 A，用它锁定格式、shape、舍入、饱和和边界正确性。
2. 实现候选 B，证明每个 block 的局部 loss 不高于 A，并在 Linear/Attention 算子输出 MSE 上跑全量自建数据。
3. 将加权与非加权 B 对比；只在隐藏分布代理数据的多数 case 不退化、总 score 改善超过 5% 时开启 sensitivity weighting。
4. 对 Weight 先开启候选 C，因为它不在高频 dynamic 路径。
5. 对 Activation/Q/K/V 分别尝试 3-base C；必须同时满足「算子 MSE 比 B 改善 >5%」和「端到端总时间保持充足余量」才采纳。
6. 5-base dynamic 只作为高成本上界对照；合成实验中它相对 3-base 的边际收益已很小。

首版的核心不是设计更多分支，而是将「给定 base 的层级精确 DP」写成一个短、无同步点、可审计的向量化内核，再用真实算子 MSE 和总时间决定是否扩展 base 候选。
