# HscCon

HscCon 是一个面向加速器的高性能 **NVFP4 → HiF4** 转换项目。项目以转换吞吐为首要目标，同时保证结果可验证、行为可复现，并为不同硬件后端保留清晰的扩展边界。

## 项目目标

- 实现 NVFP4 到 HiF4 的正确转换，明确位布局、缩放因子、舍入、饱和与特殊值语义。
- 针对连续张量和常见分块布局提供高吞吐内核，减少中间缓冲区和额外访存。
- 建立标量参考实现，作为优化内核的正确性基线。
- 建立可复现的基准测试，持续跟踪吞吐、延迟、有效带宽和转换误差。
- 采用分层后端设计，使 CPU、CUDA 及其他加速器实现可以独立演进。

## 设计原则

1. **正确性先行**：先冻结格式规范和参考结果，再开展向量化、融合及硬件专用优化。
2. **端到端性能优先**：优化真实数据路径，而不是仅追求微基准中的单条指令性能。
3. **零额外搬运**：优先原位友好的数据流、合并访存和最少的格式重排。
4. **可测量**：任何性能优化都应附带基准数据，并与参考实现进行逐项校验。
5. **渐进式优化**：保持可移植基线，使用后端特性探测选择最快的可用实现。

## 计划中的代码结构

```text
include/       公共 API 与格式定义
src/reference/ 标量参考实现
src/cpu/       CPU 向量化内核
src/cuda/      CUDA 内核
tests/         单元、边界与差分测试
benchmarks/    微基准和端到端基准
docs/          格式规范与设计文档
```

## 验收标准

转换实现进入优化阶段前，需要先具备以下测试覆盖：

- 全部可枚举的 4 位编码组合；
- 零值、符号边界、最大/最小有限值、溢出与下溢；
- 不同缩放因子、分块边界、非对齐地址和尾部元素；
- 参考实现与各优化后端之间的差分测试；
- 固定硬件、数据规模和分布下可复现的性能基线。

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

已建立比赛要求的六函数接口、NVFP4 反量化辅助函数和基础契约测试。量化函数当前显式抛出 `NotImplementedError`，将在后续比赛资料明确算法与评分约束后逐步实现；在格式契约冻结前，不对数值行为作隐式假设。
