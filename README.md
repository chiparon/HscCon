# HscCon

HscCon 是一个面向算子输出精度与转换效率的 **NVFP4 → HiF4** 量化项目。当前工作聚焦 Linear 和 Attention 的数据变换、校准统计及合法 HiF4 参数搜索，不涉及芯片架构设计。

## 项目目标

- 实现 NVFP4 到 HiF4 的正确转换，明确位布局、缩放因子、舍入、饱和与特殊值语义。
- 优化 Linear 与 Attention 的算子输出 MSE，而不只优化单张量重构误差。
- 使用校准集提取可移植统计量，并严格隔离测试数据。
- 建立可复现的基准与官方格式自检，持续跟踪精度、延迟和最差用例。
- 保持六个提交接口为单文件、固定复杂度且易于审计。

## 设计原则

1. **正确性先行**：先冻结格式规范和参考结果，再开展算法与向量化优化。
2. **算子目标优先**：直接评估 `A @ W.T` 与 Attention 输出 MSE。
3. **信息边界清晰**：校准与动态量化不访问测试集，也不以 `A @ W` 反推激活量化。
4. **可测量**：任何性能优化都应附带基准数据，并与参考实现进行逐项校验。
5. **渐进式优化**：保持可移植基线，使用后端特性探测选择最快的可用实现。

## 验收标准

版本发布前需要具备以下测试覆盖：

- 全部可枚举的 4 位编码组合；
- 零值、符号边界、最大/最小有限值、溢出与下溢；
- 不同缩放因子、分块边界、非对齐地址和尾部元素；
- 当前根版本与候选版本的配对算子级差分测试；
- 固定数据规模、分布和随机种子下可复现的性能基线。

## 提交接口

比赛提交入口为仓库根目录的 `solution.py`，公开接口固定为：

- Linear：`hif4_calibration_and_quantize_weight`、`hif4_dynamic_quantize_activation`；
- Attention：`hif4_calibration_attention`、`hif4_dynamic_quantize_q`、`hif4_dynamic_quantize_k`、`hif4_dynamic_quantize_v`。

输入由 NVFP4 carrier 和每 16 元素的 block scale 组成。HiF4 输出按每 64 元素分块，包含 `scale_factor`、`scale_lv2`、`scale_lv3`、`sign` 和 `mant`。反量化关系为：

```text
x_hat = sign * mant * scale_lv3 * scale_lv2 * scale_factor
```

calibration state 必须采用可移植的纯数据结构，不依赖可调用对象、自定义对象或外部可变状态。

赛题要求将根目录的 `solution.py` 直接打包为 `solution.zip`。仓库提供可复现的打包工具：

```bash
python tools/package_solution.py
```

判题流程、合法值域、性能门槛和 MSE 评分目标整理在 [`docs/competition_flow.md`](docs/competition_flow.md)。

数据集由多组 Linear（单份 Weight、校准集、测试集）和 Attention（Q/K/V 各自的校准集、测试集）组成。只有测试集参与 MSE 评分；单次提交全部用例共享七分钟总运行时间，单个用例没有独立限时。

> [!WARNING]
> 严禁计算或变相计算 `A @ W`，并利用其拟合、搜索或反推出 `Q(A)`；该行为经代码审核发现将按弃赛处理。动态量化实现必须保持可审计的信息边界。在线推理速度也是后续阶段的正式优化目标，方案设计应优先采用固定复杂度、可向量化且无设备同步的动态路径。

## 当前状态

已发布六函数 v2。Linear 使用 calibration-derived RMS Smooth、block 内 scale 收缩与每 64 通道正交 Hadamard；Attention 对稳定的 GQA 使用 reciprocal Smooth-QK + H64，并以校准统计门控回退；V 在同一门控下使用 guarded 三-base 搜索。相对此前根版本的公开真实样例，Linear、full Attention、causal Attention 的算子 MSE 宏平均分别改善 75.03%、39.07%、37.74%，最差 case 改善 35.61%，本地整套量化时间约为 1.51 倍。官方格式自检 22/22 通过，完整实验见 [`docs/v2_fusion_report.md`](docs/v2_fusion_report.md)。
