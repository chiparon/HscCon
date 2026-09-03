# v2 独立基准审计

## 审计边界

本报告的 control 始终是仓库当前根目录 `solution.py`（v1），与 S8、线上
提交器或历史第三方实现无关。候选收益定义为：

```text
gain = 100 * (MSE_v1 - MSE_candidate) / MSE_v1
```

因此正值只表示候选算子 MSE 更低。所有候选使用相同 public mini sample、
相同 BF16 NVFP4 反量化结果和相同算子实现做配对比较。

审计时文件 SHA256 为：

- control: `EF3543ED40AF187F3FB23897CAB5E239B4A08C7A9D0910A82825E1168BEA8E22`
- Linear: `CED85D369A79DA4DA39ECB4DB7535743190981DD50CAD4ABE30D047665DD6DAC`
- Attention: `2531EE5BA3EAAC75DD6627BF55A6BA7460E3912BC2A798A66EA6C867BC215764`
- Multi-base: `523FE3846BBC7E119AEE76FC73C34DE63BADBA8BA348223EC27145F131C46341`

## Comparator 修正与验证

`tools/hif4_official_sample_benchmark.py` 已修正/扩展：

1. 接收官方顶层 group dict、Attention `{"q","k","v"}` sample dict，
   同时兼容 list/tuple 或 `quant[_float]`/`scale[_float]` pair dict；
2. NVFP4 的乘积先舍入 BF16 再转 FP32，匹配根实现的输入解释；
3. Linear 评分严格计算 `A @ W.T`；
4. Q/K/V 按 `[tokens, heads * head_dim]` 解释，GQA 使用
   `repeat_interleave(Hq/Hkv)` 映射 KV head；
5. full 和 causal 分开评分，causal mask 为 PyTorch SDPA 的 upper-left
   `key_position <= query_position`；
6. 手写 full/causal GQA 与 `torch.nn.functional.scaled_dot_product_attention`
   的最大绝对差分别为 `2.38e-7`、`2.98e-7`；
7. 对每个 case 同时报 MSE/NMSE。两者的相对收益理论上相同，实测最大差
   不超过 `7.71e-6` 个百分点，排除了 gain 方向或归一化错误；
8. 计时先 warm-up，再以 seed `20260903` 随机打乱 case，并在每个调用内
   随机交错 control/candidate；报告 3 次完整 suite 的中位数；
9. calibration 只收到 calibration payload，测试 payload 直到 state 构造
   完成后才进入 dynamic 路径；state 通过 CPU/有限值/深度/节点/类型检查，
   且未与 calibration 输入 tensor 共用 storage；所有 HiF4 输出通过 shape、
   E6M2、lv2/lv3、sign、mant 合法性检查。

本地 PyTorch 为 2.5.1、CPU 8 threads。时间仅用于候选间相对容量判断，不能
直接换算服务器的 144 秒。

## Public mini sample 独立结果

下表每一列都是相对当前根 v1 的 case-level MSE 宏平均收益。overall 平均
15 个独立评分 case（5 Linear、5 full、5 causal），没有用 tensor 数量伪造
样本量。

| 排名 | 候选 | Linear | Attention full | Attention causal | Overall | 最差 case | 时间比 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Attention-aware | 0.000% | **32.246%** | **31.108%** | **21.118%** | 0.000% | 1.547x |
| 2 | Linear-aware | **28.260%** | 0.000% | 0.000% | **9.420%** | 0.000% | 1.484x |
| 3 | Multi-base | 7.188% | 7.431% | 7.367% | **7.329%** | 3.556% | 2.010x |

三个候选在 public 宏平均定义上都真正超过当前根版本 5%，且 gain 方向已经
由 MSE 和 NMSE 两条等价路径交叉验证。Linear-aware 的 5 个 Linear case
全部超过 24%；Attention-aware 的 10 个 Attention case 全部超过 29%。
Multi-base 虽然宏平均过线，但有一个 Linear case 只有 3.556%，且时间比已
超过 2x，证据明显弱于两个专用方向。

原始报告：

- `workbench/audit_v2_linear.json`
- `workbench/audit_v2_attention.json`
- `workbench/audit_v2_multibase.json`

## 现有 synthetic suite 泛化复核

固定 seed `20260903`、default profile 的独立结果如下：

- Linear-aware 的三个 Linear case 为 `+8.226%/+2.860%/+4.439%`，宏平均
  `+5.175%`，但只有 1/3 case 超过 5%；
- Attention-aware 的 MHA/GQA/outlier 为 `-14.410%/+33.568%/-12.532%`，
  宏平均仅 `+2.209%`，出现两个明显回退；
- Multi-base 的三个 Linear case 为 `+20.178%/+2.836%/+7.467%`，三个
  Attention case 为 `-0.776%/+1.459%/+0.895%`，六 case 宏平均约
  `+5.343%`，但只有 2/6 case 超过 5%。

这组 synthetic Attention 的 Q/K/V 是相互独立随机生成的，与投影后按 head
相关的真实 Attention 分布并不等价；它不能推翻 public 结果，但证明
Smooth+Hadamard 不具备无条件不回退性质。相应原始报告为
`workbench/audit_synth_{linear,attention,multibase}.json`。

## 结论与融合建议

1. **采纳 Linear-aware 方向。** 它在真实 public Linear 的每个 case 都远超
   5%，且与 Attention 函数不重叠；synthetic 宏平均也刚好过线。
2. **有条件采纳 Attention-aware 方向。** public 的 full/causal 证据很强，
   但通用随机 MHA/outlier 会回退。融合版应保留校准分布门控或回退到 v1 的
   开关，并在未参与选择的结构化 MHA/GQA seed 上复核后发布。
3. **不建议再叠加当前 Multi-base。** 它的专用收益远低于前两者、存在不足
   5% 的 case，且独立运行已达 2.01x；其收益很可能与 Linear base search
   重叠。若融合后单独消融仍额外提升超过 5%，再考虑加入。
4. 最优融合排序不是把三个实现机械相加，而是 Linear 两接口取
   Linear-aware，Attention 四接口取经过泛化门控的 Attention-aware；最终
   融合结果必须重新相对根 v1 运行同一 comparator，不能把两个子代理各自
   的收益直接相加。
