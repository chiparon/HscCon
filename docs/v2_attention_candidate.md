# v2 Attention 适配候选

## 结论

推荐将 `workbench/solution_v2_attention_candidate.py` 中的 **按 head
Smooth-QK + 相同 64 维正交 Hadamard + 固定 base DP** 合入 v2。它在公开
mini sample 上相对当前根目录 v1 的 full/causal SDPA MSE 分别下降
33.29%/31.43%，显著超过 5% 采纳线。

对端二阶矩加权 DP 和三 base 搜索已经实现并完成消融，但没有打开：它在
公开数据上的 full MSE 只比已采纳路径再下降约 1.51%，causal MSE 反而回退
约 0.11%，且动态路径明显变慢，不满足“方向收益 >5%”门槛。

## 已确认的数据契约

公开 `self_check.py` 和 `data/attn.pt` 均确认当前 Attention layout 是：

```text
q_quant: [seq_len, q_num_heads * head_dim]
k_quant: [seq_len, kv_num_heads * head_dim]
v_quant: [seq_len, kv_num_heads * head_dim]
calib sample: {q: [quant, scale], k: [quant, scale], v: [quant, scale]}
```

mini sample 为 GQA：`q_num_heads=16`、`kv_num_heads=2`、`head_dim=256`，
五份 calibration 和五份 test，序列长度为 10/128/512/1024/1024。
候选同时兼容仓库 benchmark 使用的六元 tuple payload，但不依赖未证实的
其他 layout。

## 算法

校准阶段只分别解码 Q/K，并按 `(head, head_dim)` 聚合二阶矩。GQA 中，
连续的 `Hq/Hkv` 个 Q head 映射到一个 KV head。对每个 KV head 得到：

```text
q2 = mean(Q_group ** 2)
k2 = mean(K ** 2)
s  = clamp((q2 / k2) ** 0.125, 1/16, 16)
```

动态阶段执行：

```text
Q' = Hadamard64(Q / s)
K' = Hadamard64(K * s)
```

同一 KV group 的 Q/K 使用同一 `s` 和同一正交变换，因此浮点点积保持：

```text
Q' @ K'.T == Q @ K.T
```

转换误差只来自随后的 HiF4 量化。Smooth 平衡成对 Q/K 的尺度；Hadamard
把每个 head 内 64 维块的局部尖峰和方差集中扩散开，适配 HiF4 的 64 元素
共享 base。V 保持 v1 路径，避免改变 Attention value 空间。

候选也计算变换后对端二阶矩，并实现了 Q/K 加权 HiF4 DP。实验版只在加权
loss 至少下降 1%、普通重构 SSE 回退不超过 0.25% 时逐 block 接受候选。
这一分支保留在代码中，可通过 state 的 `use_weighted/search_radius` 做受控
实验；当前默认 state 明确关闭。

## 质量结果

公开 mini sample（真实 payload，五个 test）：

| 路径 | Full SDPA mean MSE | Causal SDPA mean MSE |
| --- | ---: | ---: |
| v1 | 0.000889810 | 0.001026341 |
| Hadamard only | 0.000627938 | 0.000787352 |
| Smooth + Hadamard（采纳） | 0.000593575 | 0.000703780 |
| Smooth + Hadamard + 加权三-base | 0.000584644 | 0.000704588 |

Smooth 在 Hadamard 之上继续将 full/causal MSE 降低 5.47%/10.62%，自身也
通过 5% 方向门槛。加权三-base 相对采纳路径不满 5%，所以拒绝。

新增的结构化 Attention 数据包含 MHA d64、GQA 4:1 d128、GQA 8:1 d256，
各自拥有独立 calibration/test；三组 full MSE 改善分别为
53.81%/27.79%/5.73%，平均 full MSE 下降约 29.4%。causal MSE 三组改善
49.94%/14.98%/4.30%，平均下降约 23.2%。

原通用随机 benchmark 中 MHA 和 outlier case 存在回退；它的 Q/K/V 相互
独立且全 hidden 维使用单调 gain，不代表投影后按 head 的典型结构。因此
保留这一结果作为风险信号，不用它替换公开 mini sample 和专用数据证据。

## 时间结果

公开 mini sample、8 CPU threads：

- v1 五次 Q/K/V dynamic 合计约 0.219 s；
- 候选五次 dynamic 合计约 0.259 s，约 1.19 倍；
- 候选 calibration 约 0.070 s；
- calibration + dynamic 总计约为 v1 的 1.51 倍。

仓库完整 default benchmark 三次交错测量为 v1 45.01 ms、候选 63.66 ms，
总耗时约 1.41 倍。按线上 v1 的 144 s 线性粗估约 203 s；这里只作为容量
估计，不能替代服务器计时。

## 验证

- 公开 `self_check.py` Attention 16/16 通过；
- MHA/GQA、head_dim 64/128/256 参数与输出 shape/值域通过；
- 随机 Q/K 上 inverse Smooth + Hadamard 的 dot-product 保持断言通过；
- 公开与合成 benchmark 均冻结 calibration state 后才访问 test；
- state 仅含有限 tensor/scalar 统计，不保存 calibration 样本或可调用对象；
- 不在算法内计算 token×token attention，也不读取 test 数据。

复现命令：

```powershell
$env:PYTHONPATH='E:\Eproject\HiFCon'
& 'C:\Users\chiparon\miniconda3\envs\inpaint\python.exe' workbench\test_v2_attention_candidate.py
& 'C:\Users\chiparon\miniconda3\envs\inpaint\python.exe' workbench\benchmark_v2_attention.py --threads 8
& 'C:\Users\chiparon\miniconda3\envs\inpaint\python.exe' workbench\benchmark_v2_attention_synthetic.py
```
