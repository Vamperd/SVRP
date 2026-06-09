# SVRP/TWCVRP 强化学习复现执行脉络

本文记录项目推进过程中有复盘价值的问题、决策和阶段性结论。后续每次沟通都会判断是否值得追加到本文：若问题影响技术路线、实验设计、环境配置、评估解释或最终报告表达，则记录；若只是一次性命令确认或重复性排错，则只在必要时合并记录。

## 记录准则

- 记录会改变实现路线的问题，例如是否继续使用 RL4CO、是否新建独立 RL 链路。
- 记录会影响实验可信度的问题，例如训练/验证/测试如何划分、是否能跨规模泛化。
- 记录会影响报告解释的问题，例如为什么 RL 可行率低、如何和 OR-Tools 指标对比。
- 记录关键环境问题，例如 CUDA/CPU PyTorch、OpenMP 绘图冲突、依赖兼容性。
- 不记录纯重复命令，除非它暴露出新的实验风险。

## 高价值问题与决策时间线

### 1. 是否复现论文中的 RL 方法并接入 `vrp_benchmark`

用户最初目标是阅读论文 PDF，依照论文中的 RL 方法复现代码，并让模型解决 `vrp_benchmark` 数据集中的路径规划问题。

阶段结论：

- 初始尝试基于 `svrpbench/models/rl`、RL4CO、TorchRL、Lightning。
- 该路线遇到多次依赖兼容问题，包括 TorchRL API 变化、RL4CO 环境缺失、PyTorch checkpoint `weights_only` 与 RNG 反序列化问题。
- 结论是：论文级 RL4CO 链路保留但不作为当前主线。

报告启示：

- 复现论文方法时，依赖版本本身会成为实验风险。
- 后续报告中应说明“为保证实验闭环，改用独立 PyTorch 链路作为 RL 对照组”。

### 2. 是否需要撤回先前所有更改

用户在多次补丁叠加后询问是否需要撤回全部更改。

阶段结论：

- 不建议全量撤回。
- 保留已跑通的独立 CVRP 链路。
- 暂停使用不稳定的 RL4CO 链路。
- 新建独立 TWCVRP/Solomon 链路，避免继续污染旧入口。

报告启示：

- 该问题标志着项目从“论文完整复现”转向“可运行、可对照、可解释的 RL 实验链路”。

### 3. CVRP 独立链路是否能使用 `vrp_benchmark` 测试

用户关心当前训练结果是否可以测试 `vrp_benchmark` 中的数据。

阶段结论：

- 新增 `svrpbench/models/rl_standalone`。
- 该链路可读取 `vrp_benchmark/real_cvrp/*.npz`，进行纯 PyTorch REINFORCE 训练和推理。
- 该链路解决 CVRP，不处理时间窗和交通扰动。

报告启示：

- CVRP 链路是最小可运行基线，用于验证“数据集 -> RL 训练 -> 路线输出”的闭环。

### 4. CUDA 环境与 PyTorch 版本问题

用户先后询问 CUDA 12.5 是否可用 `cu124` 命令安装，以及为什么 `torch.__version__` 显示 `2.2.2+cpu`。

阶段结论：

- `2.2.2+cpu` 表示当前环境安装的是 CPU 版 PyTorch，不是代码未调用 GPU。
- 若 `nvidia-smi` 正常，进入正确 conda CUDA 环境或安装 CUDA 版 PyTorch 后，训练脚本会显示 `device=cuda`。
- 用户后续确认问题原因是没有进入 CUDA 环境。

报告启示：

- 环境章节应区分“conda 环境是否激活”和“PyTorch wheel 是否带 CUDA”。
- 大规模训练建议使用 GPU；若显存不足，降低 `batch_size`。

### 5. OpenMP 绘图冲突是否需要修正

用户在 `solve.py --plot` 和 `plot_results.py` 中遇到 `libomp.dll` 与 `libiomp5md.dll` 冲突。

阶段结论：

- JSON 推理结果不依赖绘图，可先不绘图。
- Matplotlib/Qt/PyTorch 在 Windows 下容易触发 OpenMP 冲突。
- 后续 TWCVRP 链路改用 SVG 绘图，避免 Matplotlib 依赖。

报告启示：

- 实验输出应优先保证 JSON/CSV 指标，图像作为辅助。

### 6. TWCVRP 与 CVRP 的核心差别

用户询问 TWCVRP 与 CVRP 的差别是否只是服务时间窗。

阶段结论：

- CVRP 主要约束是容量和路径成本。
- TWCVRP 在 CVRP 基础上增加 ready time、due date、service time。
- 早到会等待，晚到会产生时间窗违约。
- 交通扰动本质上影响旅行时间，从而影响到达时间和时间窗可行性。

报告启示：

- 后续对照报告应把“容量约束”和“时间窗约束”分开解释。

### 7. 如何参照 `benchmark_report.pdf` 形成 RL 对照组

用户提供 `benchmark_report.pdf`，希望 RL 能和报告实现形成对照组，并解决其中提到的静态、交通扰动、时间窗问题。

阶段结论：

- 报告主线是 Solomon/Homberger TWCVRP。
- 新增 `svrpbench/models/rl_solomon_tw`，直接读取本地 `solomon/100/200/400/600/800/1000`。
- 实现静态评估、交通扰动训练、hybrid 评估、traffic 评估。
- 输出指标对齐报告：`total_cost`、`waiting_time`、`cvr`、`feasibility`、`runtime`、`robustness_std` 等。

报告启示：

- RL 对照组不要求达到 OR-Tools 最优，但必须输出同口径指标。

### 8. 是否必须预生成交通扰动矩阵

用户询问 `prepare_traffic.py` 是否是测试交通扰动效果的必要步骤。

阶段结论：

- 不必须。
- 训练和评估脚本默认按固定 seed 在线生成交通扰动矩阵。
- `prepare_traffic.py --store full` 只用于审计或复现实验矩阵，1000 规模会占用较多磁盘。

报告启示：

- 交通扰动实验应说明扰动是合成的、seed 固定、可在线生成。

### 9. 是否应该划分训练集、验证集和测试集

用户询问是否应该单独使用部分数据训练、部分数据检测。

阶段结论：

- 必须划分，否则测试结果无法说明泛化能力。
- CVRP 使用 `.npz` 内 instance index 划分。
- TWCVRP 使用 Solomon 文件级划分。
- 训练集更新模型，验证集选择 checkpoint，测试集只用于最终报告。

报告启示：

- 最终报告应明确 train/val/test，不应使用训练集结果作为最终性能。

### 10. 是否能训练一个通用网络求解任意规模

用户询问是否能设计单独网络结构，使训练后可求解 `100/200/400/1000` 等任意规模。

阶段结论：

- 当前 pointer policy 结构本身支持可变客户数量，因为对客户共享编码器并逐步 attention 选择。
- 但训练必须覆盖目标规模范围，否则从 100 直接泛化到 1000 容易退化。
- 新增 `data_splits/universal_v1`：
  - `100/200/400` 为主训练规模。
  - `600/800/1000` 加入少量训练文件做大规模适应。
  - 所有规模保留 val/test。
- 修改 `train.py` 支持 mixed-size training；checkpoint 保存 `supported_sizes`。

报告启示：

- 通用模型是本项目后续主线，可作为跨规模泛化实验。

### 11. 当前静态 100 训练结果是否合理

用户给出 `static_100.pt` 训练日志，询问是否合理。

阶段结论：

- 日志显示训练流程正常，GPU 已启用，reward 为负数正常。
- 训练集 cost 有下降趋势，但验证集可行率长期停留在约 `0.38`。
- 默认 `epochs=100` 且 `steps_per_epoch=1`，实际只有 100 次梯度更新，对 REINFORCE 偏少。
- 建议增加 `steps_per_epoch`，并优先使用 best checkpoint。

报告启示：

- 初版 RL 是流程闭环，不代表已经达到可用质量。

### 12. 为什么 RL 成功率不能提升，当前策略是什么

用户拿报告中的 OR-Tools 指标与 RL 结果对比，询问为什么成功率不提升，以及当前策略是什么。

阶段结论：

- 当前策略是轻量 pointer policy：模型输出客户访问顺序，decoder 再按容量和时间窗切成车辆路线。
- 约束主要由 decoder 兜底，不是模型直接求约束优化。
- 当前 best checkpoint 选择主要看验证成本，可能选到成本较低但不可行的模型。
- 成功率低的主要原因包括训练步数少、车辆数超限惩罚不足、模型状态与 decoder 切车逻辑不完全一致。

报告启示：

- 和 OR-Tools 对比时，必须先比较可行率和 CVR，再比较成本。
- 后续优化方向应优先提高 feasibility，而不是单纯降低 total_cost。

### 13. 如何评估 `results` 中当前训练后的测试结果

用户询问如何评估 `svrpbench/models/rl_solomon_tw/results` 中当前测试结果。

阶段结论：

- 当前主要测试文件是 `static_100_test.csv/json`。
- 结果显示：
  - `static_rl` 测试集 9 个实例，可行率约 `22.22%`，`CVR=23.56%`。
  - `hybrid_rl` 交通扰动下可行率为 `0%`，`CVR=24.39%`。
  - 时间窗和容量违约不严重，主要问题是 `vehicles_excess` 约 `23.56`。
- 当前 RL 使用约 `47.89` 条路线，而 Solomon 100 可用车辆数是 `25`，所以多数实例不可行。

报告启示：

- 当前模型失败主要不是迟到，而是为了避免迟到频繁开新车，导致车辆数超限。
- 后续训练和 decoder 优化应重点压低 `vehicles_excess`。

### 14. 为什么增大惩罚后可行率仍然不提升，是否需要重构策略

用户使用更强惩罚重新训练：

```text
vehicle_penalty=5000
time_window_penalty=5000
capacity_penalty=5000
late_penalty=50
epochs=300
steps_per_epoch=20
```

训练后 `val_feas` 仍主要停留在 `0.38-0.50`，说明问题不是单纯 penalty 太小。

阶段结论：

- 当前主要矛盾是结构性问题：旧 decoder 遇到迟到会开新车，容易造成 `vehicles_excess`。
- 模型只输出客户顺序，没有直接学习有限车辆内的插入修复。
- best checkpoint 原先按 `val_total_cost` 选择，可能偏向低成本但不可行的模型。
- 因此需要转向可行性优先 v2：
  - checkpoint 选择优先 `val_feasibility`。
  - reward 增加 `feasible_bonus` 和 `infeasible_penalty`。
  - decoder 改为 `strict_insert`，最多使用允许车辆数。
  - 模型加入动态可行性特征。
  - 训练前加入启发式 imitation warm-start。

报告启示：

- 该问题是从“调参”进入“算法结构重构”的关键转折点。
- 报告中应强调：TWCVRP 的 RL 难点不是输出路径本身，而是在容量、时间窗、车辆数三类硬约束之间保持可行。

## 当前推荐报告叙事

1. 首先说明原论文/RL4CO 链路因依赖兼容风险暂停。
2. 然后说明建立独立 PyTorch CVRP 链路验证基本训练闭环。
3. 再说明根据报告主问题切换到 Solomon/TWCVRP。
4. 介绍静态、hybrid、traffic 三类评估。
5. 说明当前 RL 对照组已经可输出同口径指标，但质量弱于 OR-Tools。
6. 重点分析失败原因：车辆数超限、REINFORCE 训练不稳定、约束主要靠 decoder。
7. 最后提出后续优化：可行性优先 checkpoint、增强车辆惩罚、改进 decoder、增加训练步数、多规模通用模型。
