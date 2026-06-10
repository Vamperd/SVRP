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

### 15. `static_100_v2_smoke_test` 说明当前方向是否正确

用户要求读取 `results/static_100_v2_smoke_test.json` 并判断 v2 方向是否正确。

阶段结论：

- v2 方向明确正确。
- 静态测试集中：
  - `static_rl feasibility=1.0`
  - `static_rl cvr=0.0`
  - `vehicles_excess=0.0`
  - `route_count≈12.67`
- 这说明 `strict_insert` 已经解决旧模型的核心问题：频繁开新车导致车辆数超限。
- 交通扰动 hybrid 中：
  - `feasibility=0.0`
  - `cvr≈2.81`
  - `time_window_violations≈2.81`
  - `vehicles_excess=0.0`
- 这说明剩余问题已经从“车辆数超限”转向“交通扰动下的时间窗鲁棒性”。

报告启示：

- 报告应把 v2 作为一次有效算法修正：从不可行路线转向静态可行路线。
- 后续实验重点应比较 `hybrid_rl` 和 `traffic_rl`，验证交通感知训练是否降低时间窗违约。
- 当前不应回退旧 decoder；应在 `strict_insert` 基础上优化交通鲁棒性和训练速度。

### 16. v2 训练速度过慢，需要紧急优化

用户反馈 100 规模训练已经跑不动，急需优化。

阶段结论：

- 性能瓶颈主要在 `strict_insert` decoder，而不是 GPU。
- 旧实现对每个客户尝试所有插入位置，并且每个候选都会完整评估所有路线，训练成本很高。
- 已实施止血优化：
  - `strict_insert` 改为只重评被插入的一条 route。
  - 新增 `--insert_top_k`，只保留最有希望的插入候选。
  - 新增 `--val_limit`，训练中减少验证集开销。
  - imitation 专家顺序加入内存缓存，避免同一实例反复生成专家顺序。

报告启示：

- v2 成功提升可行性后，计算复杂度成为新的工程瓶颈。
- 报告中可以说明：可行性优先解码器牺牲了一定训练速度，后续通过候选剪枝和增量评分进行优化。

### 17. 将 TWCVRP 说明文档收束到当前 v2 主线

用户要求优化 `svrpbench/models/rl_solomon_tw/README_TW_RL.md`，只保留当前策略相关内容，同时保证流程完整。

阶段结论：
- 当前主线明确为独立 PyTorch TWCVRP 链路。
- 旧 RL4CO/TorchRL/Lightning 路线、旧 `greedy_split` 主流程和早期固定规模命令不再作为操作主线。
- 文档应围绕 v2 可行性优先策略组织：`strict_insert` decoder、imitation warm-start、可行率优先 checkpoint、`insert_top_k`/`val_limit` 加速、静态与交通扰动对照评估。
- 文档中的命令应优先服务当前瓶颈：100 规模训练过慢，因此先给快速确认命令，再逐步扩大训练。

报告启示：
- 这是工程链路从“功能堆叠”转向“可复现实验流程”的整理节点。
- 最终报告可说明：当前 RL 对照组已经形成稳定的静态 TWCVRP 可行解生成流程，后续实验重点是交通扰动鲁棒性和训练速度。

### 18. 加速后的 v2 quick 是否明显损害性能

用户完成 `static_100_v2_quick.pt` 训练后询问：训练速度已经明显变快，`results/static_100_v2_quick.json` 是否说明前面的加速修改大幅影响了性能。

阶段结论：
- quick 结果没有破坏最关键的静态可行性：
  - `static_rl feasibility=1.0`
  - `static_rl cvr=0.0`
  - `vehicles_excess=0.0`
  - `time_window_violations=0.0`
- 相比 `static_100_v2_smoke_test.json`，quick 静态总成本从约 `3052.41` 降到 `2827.98`，总距离从约 `2353.75` 降到 `1687.82`，但等待时间从约 `698.66` 升到 `1140.16`，路线数从约 `12.67` 升到 `14.33`。
- quick 的 hybrid 指标表面上也改善：`cvr` 从约 `2.81` 降到 `1.64`，`late_minutes` 从约 `19.30` 降到 `4.42`，但 quick 文件使用的扰动采样数较小，需要用 `mc_samples=30` 再做同口径确认。

报告启示：
- `strict_insert` 的候选剪枝和训练降采样没有明显破坏核心可行性，说明当前加速方向可继续保留。
- 报告中应区分“静态可行性已经稳定”和“交通扰动鲁棒性仍需同口径 MC 评估”。

### 19. 如何从 quick 命令扩大到更真实的训练规模

用户询问如何修改当前 quick 训练命令，使训练规模更大，更能体现真实大规模训练后的效果。

阶段结论：
- 当前 quick 命令主要用于快速验证，实际训练量很小：`epochs=30`、`steps_per_epoch=3`、`batch_size=2`，总采样量和优化步数都偏少。
- 更真实的训练不应只增加 `epochs`，还应同时调整：
  - `steps_per_epoch`：决定每个 epoch 的优化步数。
  - `batch_size`：影响梯度稳定性。
  - `insert_top_k`：决定 `strict_insert` 搜索质量。
  - `imitation_epochs`：让模型更稳定地进入可行区域。
  - `val_limit`：quick 可限制验证集，正式训练应使用完整验证集或更大的验证子集。
- 建议训练层级从 `quick` -> `medium` -> `full_100` -> `universal/mixed-size` 逐步扩大，避免直接跳到 600/800/1000 导致训练时间失控。

报告启示：
- 后续报告应明确区分快速可行性验证和正式训练结果。
- 最终性能表应使用 full 或 medium checkpoint 在独立 test split 上评估，而不是使用 quick 训练日志作为最终结论。

### 20. `static_100_v2_full_test` 正式静态模型测试结果评估

用户要求检验 `results/static_100_v2_full_test.json`，并综合考量成功率与用时评分。

阶段结论：
- `static_rl` 在 100 规模 test split 上表现稳定：
  - `instances=9`
  - `feasibility=1.0`
  - `cvr=0.0`
  - `vehicles_excess=0.0`
  - `time_window_violations=0.0`
  - `total_cost≈2292.50`
  - `route_count≈10.44`
  - `runtime≈0.000301s`
- 相比 quick 结果，full 静态模型在保持 100% 可行的同时，进一步降低了 `total_cost`、`waiting_time` 和 `route_count`，说明扩大训练是有效的。
- `hybrid_rl` 仍然较弱：
  - `feasibility≈0.111`
  - `cvr≈2.259`
  - `time_window_violations≈2.259`
  - `vehicles_excess=0.0`
  - 失败来源已经明确集中在交通扰动下的时间窗违约，而不是车辆数或容量。

报告启示：
- `static_100_v2_full_test` 可以作为静态 TWCVRP RL 对照组的正式结果。
- 若报告讨论交通扰动或鲁棒性，不能只用该静态模型，应继续训练并评估 `traffic_rl` checkpoint。
- 求解用时极短，但当前 runtime 是推理/评估阶段时间，不包含训练成本，和传统算法比较时应注明口径。

### 21. 当前平均总成本的计算口径

用户追问当前结果中的“平均总成本”是如何考量的。

阶段结论：
- 当前 `evaluator.py` 中单实例 `total_cost` 的定义为：
  - `total_cost = total_travel_time + waiting_time + late_minutes`
- 其中：
  - `total_travel_time` 为路径行驶时间；静态场景下等于欧氏距离时间。
  - `waiting_time` 为早到后等待到时间窗开始的等待时间。
  - `late_minutes` 为迟到分钟数，评估时直接相加，不乘训练惩罚系数。
- 当前 `service_time` 被统计但没有计入 `total_cost`。
- 车辆数超限、容量违约、漏访、重复访问等不直接加进 `total_cost`，而是通过 `cvr` 和 `feasibility` 单独评价。
- 聚合结果中的平均总成本是 test split 中所有实例 `total_cost` 的算术平均。

报告启示：
- 当前 `total_cost` 是项目内部 RL 评估口径，不能默认与外部报告中的“平均总成本”直接比较。
- 若需要和外部基准公平比较，应统一成本函数，例如明确是否纳入服务时间、车辆固定成本、迟到惩罚权重和完整求解时间。

### 22. 多规模 universal 训练前必须先生成文件级 split

用户执行 universal 全规模训练命令时报错：

```text
FileNotFoundError: No TWCVRP data found under data_splits\universal_v1\train for size=100.
```

阶段结论：
- 本地检查显示 `svrpbench/models/rl_solomon_tw` 下没有 `data_splits/universal_v1` 目录，也没有 `split_manifest.json`。
- `train.py --split_root data_splits\universal_v1` 会查找：
  - `data_splits/universal_v1/train/100/*.txt`
  - `data_splits/universal_v1/val/100/*.txt`
  - 以及其它规模目录。
- 当前错误不是模型训练失败，而是文件级划分尚未生成或路径指向错误。
- 修复方式是先运行 `create_universal_split.py` 复制 Solomon 文件并生成 manifest，再运行 `inspect_split.py` 检查数量。

报告启示：
- 多规模通用模型实验必须把数据划分生成步骤写入实验流程，否则训练命令不可复现。
- `universal_v1` 是文件级划分，不是直接读取原始 `solomon` 根目录。

### 23. universal 全规模训练是否应该扩大训练规模

用户询问当前 universal 全规模命令是否应该继续扩大训练规模：

```text
sizes=100 200 400 600 800 1000
epochs=30
steps_per_epoch=6
batch_size=1
insert_top_k=10
imitation_epochs=3
```

阶段结论：
- 该命令已经覆盖所有目标规模，但训练强度仍属于 stage1/quick，不适合作为最终通用模型结果。
- `steps_per_epoch=6` 在 balanced sampling 下约等于每个规模每 epoch 被采样一次，`30*6=180` 次优化步对六个规模而言偏少。
- 不建议立即扩大 `batch_size` 或直接提高 `insert_top_k` 到很大，因为 `strict_insert` 在 600/800/1000 上主要受 CPU/Python 插入评估限制。
- 更合理的扩展顺序是：
  1. 先跑当前 stage1，确认全规模流程可闭环。
  2. 增加 `epochs` 和 `steps_per_epoch`，保持 `batch_size=1`、`insert_top_k=10`。
  3. 稳定后再提高 `insert_top_k` 或分阶段训练 `100/200/400` 与 `600/800/1000`。

报告启示：
- universal 模型最终结果不应使用 stage1 quick checkpoint。
- 报告中应把 universal 训练分为流程验证、阶段训练和正式评估三层，避免把快速实验误读为充分训练。

### 24. 新增 PDF 同口径评估程序并对比 100 专用模型与 universal 模型

用户要求新增依照 `metrics_formula.md` 的 PDF/OR-Tools 同口径评估程序，用于评估：

```text
checkpoints/static_100_v2_full_best.pt
checkpoints/tw_universal_all_full_best.pt
```

阶段结论：
- 新增 `benchmark_evaluate.py`，不改变训练 reward，不影响原 `evaluate.py`。
- PDF 口径指标包括：
  - `avg_cost = total_travel_time + waiting_time`
  - `avg_cvr = 100 * (time_window_violations + capacity_violations) / num_customers`
  - `feasibility_rate` 不统计车辆超限，只按漏访、重复、时间窗、容量判断。
  - `avg_solver_runtime_s` 包含模型 forward、decoder、route evaluation。
- 已生成三组静态评估结果：
  - `results/benchmark_common_clean_100.csv/json`
  - `results/benchmark_static100_default_test.csv/json`
  - `results/benchmark_universal100_test.csv/json`
- 在三组 100 规模静态评估中，两个模型均达到：
  - `avg_cvr=0.0`
  - `feasibility_rate=1.0`
  - `avg_vehicles_excess=0.0`
- 严格共同干净测试集 `c202/c203` 上：
  - 100 专用模型 `avg_cost≈3904.50`
  - universal 模型 `avg_cost≈4258.80`
- 两个各自 9 实例测试集上，100 专用模型成本也略低；universal 模型推理时间略低但差距很小。

报告启示：
- 在 100 规模静态问题上，100 专用模型和 universal 模型都已达到可行性指标；区别主要体现在成本。
- 当前结果支持如下表述：专用训练模型在 100 静态成本上略优，universal 模型牺牲少量 100 规模成本以换取跨规模适应能力。
- 严格公平比较应优先引用 `common_clean_100`；各自 split 结果只能作为补充说明。

### 25. 使用 100 专用模型和 universal 模型评估 200 规模测试集

用户询问如何使用 `static_100_v2_full_best.pt` 和 `tw_universal_all_full_best.pt` 对 200 规模测试集合进行评估。

阶段结论：
- `static_100_v2_full_best.pt` 的 checkpoint 元数据只声明支持 `100`，用于 200 时属于跨规模泛化测试，需要显式添加 `--allow_unseen_size`。
- `tw_universal_all_full_best.pt` 声明支持 `100/200/400/600/800/1000`，可直接评估 200。
- 推荐使用 `benchmark_evaluate.py` 的 PDF 同口径评估，并优先使用 `data_splits/universal_v1/test/200` 作为 200 测试集。

报告启示：
- 200 规模对比应明确区分“100 专用模型的跨规模泛化表现”和“universal 模型的目标规模表现”。
- 如果 100 专用模型在 200 上表现较差，这是合理现象，不应简单视为训练失败。

### 26. `benchmark_200_test` PDF 同口径结果分析

用户要求分析 `results/benchmark_200_test.csv` 中 200 规模静态测试结果。

阶段结论：
- 测试集为 `universal_v1_test`，规模 `200`，实例数 `9`，模式为 `static`。
- 两个模型均达到：
  - `avg_cvr=0.0`
  - `feasibility_rate=1.0`
  - `avg_vehicles_excess=0.0`
  - `avg_route_count≈19.22`
- 100 专用模型跨规模评估结果：
  - `avg_cost≈8219.14`
  - `single_customer_cost≈41.10`
  - `avg_waiting≈3096.68`
  - `avg_solver_runtime_s≈0.383`
- universal 模型结果：
  - `avg_cost≈8667.88`
  - `single_customer_cost≈43.34`
  - `avg_waiting≈3633.13`
  - `avg_solver_runtime_s≈0.375`
- 单实例上，100 专用模型在 `6/9` 个实例成本更低，universal 模型在 `3/9` 个实例成本更低；最大差异来自 `C2_2_1`，universal 成本明显更高。

报告启示：
- 200 静态场景下两个模型都能保持可行性，说明当前 `strict_insert` 可行性优先解码对跨规模有效。
- 当前 universal 模型在 200 成本上没有优于 100 专用模型，说明它虽具备跨规模能力，但训练强度或多规模权衡仍需优化。
- 不能简单宣称 universal 必然优于单规模模型；更合理的表述是 universal 提供跨规模统一求解能力，但在部分规模上可能牺牲单规模成本。

### 27. 100-1000 全规模 PDF 同口径静态测试汇总

用户完成 400/600/800/1000 规模评估后，要求按报告表格形式统计并说明结果。

阶段结论：
- 已汇总 `100/200/400/600/800/1000` 六个规模的 PDF 同口径静态 TWCVRP 测试结果。
- 输出文件：
  - `svrpbench/models/rl_solomon_tw/results/benchmark_all_sizes_summary.md`
- 两个模型在所有规模上均达到：
  - `CVR=0.0%`
  - `feasibility_rate=100%`
  - `avg_vehicles_excess=0.0`
- `static100` 在所有规模上的 `avg_cost` 均低于 `universal`：
  - 100: 低约 `6.2%`
  - 200: 低约 `5.5%`
  - 400: 低约 `3.6%`
  - 600: 低约 `0.4%`
  - 800: 低约 `0.5%`
  - 1000: 低约 `2.7%`
- 随规模增大，等待占比显著升高，600-1000 规模接近或超过 `48%`，表明大规模 TWCVRP 的主要成本压力来自时间窗等待。
- 求解时间随规模上升，但仍保持秒级，1000 规模约 `2.09-2.11s`。

报告启示：
- 当前可行性优先解码器已经实现了跨规模静态 TWCVRP 可行性闭环。
- universal 模型的价值应表述为“单 checkpoint 覆盖多规模”，而不是当前阶段成本优于单规模/专用模型。
- 后续若要提升 universal，应重点降低等待时间和路线组织成本，而不是继续只追求可行率。

### 28. 源 PDF/OR-Tools 成本计算与当前 RL 评估口径差异

用户提供源 PDF 对应算法目录 `Solomon_partner/Solomon`，要求分辨其成本计算并对比两边算法优劣。

阶段结论：
- 源算法核心文件为 `svrp_ortools_twcvrp/src/evaluator.py`、`experiment_runner.py`、`ortools_twcvrp_solver.py`。
- 源算法报告成本为：
  - `total_cost = total_travel_time + waiting_time`
  - `CVR = 100 * (tw_violations + capacity_violations) / num_customers`
  - feasibility 额外检查漏访、重复访问、时间窗和容量。
- 源算法静态时间矩阵来自欧氏距离并四舍五入为整数，且 evaluator 中不累计 service time。
- 当前 `benchmark_evaluate.py` 的输出指标贴近源 PDF 公式，但底层路线时间推进仍使用当前 Solomon 评估器：
  - 使用浮点欧氏距离。
  - 读取并累计 Solomon service time，影响后续到达时间、等待和时间窗违约。
- 因此目前只能称为“PDF-style 指标”，不能称为与源 PDF/OR-Tools 完全同口径。

报告启示：
- 报告中可比较 `CVR=0`、`可行率=100%` 等可行性结果，但成本数值不能直接宣称显著优于源 PDF 表格。
- 若要完全公平比较，应新增 source-compatible evaluator：整数化 travel matrix、忽略 service time 或与源算法一致处理 service time，并在同一批实例上同时跑 OR-Tools 与 RL。

## 当前推荐报告叙事

1. 首先说明原论文/RL4CO 链路因依赖兼容风险暂停。
2. 然后说明建立独立 PyTorch CVRP 链路验证基本训练闭环。
3. 再说明根据报告主问题切换到 Solomon/TWCVRP。
4. 介绍静态、hybrid、traffic 三类评估。
5. 说明当前 RL 对照组已经可输出同口径指标，但质量弱于 OR-Tools。
6. 重点分析失败原因：车辆数超限、REINFORCE 训练不稳定、约束主要靠 decoder。
7. 最后提出后续优化：可行性优先 checkpoint、增强车辆惩罚、改进 decoder、增加训练步数、多规模通用模型。

### 29. 修正 PDF 同口径评估并澄清“平均客户成本”上升

用户进一步明确：需要与原 PDF 进行直接对比，因此评估算法必须完全向 PDF/OR-Tools 代码看齐；同时此前讨论中的“平均服务时间上升”实际指的是 `single_customer_cost = avg_cost / num_customers`，即平均客户成本上升。

阶段结论：
- 已将 `benchmark_evaluate.py` 的报告评估路径改为 `pdf_compatible`：
  - 静态矩阵使用欧氏距离并四舍五入为整数；
  - 时间推进不再加入 Solomon `service_time`；
  - `avg_cost = total_travel_time + waiting_time`；
  - `CVR = 100 * (time_window_violations + capacity_violations) / num_customers`；
  - 可行率按漏访、重复访问、时间窗、容量判断，车辆超限继续作为额外诊断字段输出。
- 新 evaluator 中 `service_time` 固定为 `0.0`，另保留 `ignored_service_time` 方便报告说明：Solomon 原始服务时间存在，但不参与 PDF 同口径评分。
- 已完成 `c202/c203` sanity check，验证 `benchmark_total_cost == total_travel_time + waiting_time`。

报告启示：
- 后续与 PDF 表格直接对比时，只能使用新的 `pdf_compatible` 结果。
- 若新口径下平均客户成本仍随规模上升，应解释为大规模时间窗等待、跨规模泛化和可行性优先 decoder 的综合影响，而不是服务时间被直接计入成本。
- 不建议覆盖 `static100`；它应保留为“100 规模训练、跨规模 zero-shot 泛化”基线。如需提升大规模平均客户成本，应另训多规模或单规模大模型。

### 30. PDF 同口径 100-1000 全规模结果分析

用户完成 `benchmark_pdf_100_test` 至 `benchmark_pdf_1000_test` 的新口径评估，要求像之前一样整理分析。

阶段结论：
- 已生成 `results/benchmark_pdf_all_sizes_summary.md`，使用完全 PDF 同口径结果。
- 两个 RL 模型在 100/200/400/600/800/1000 全部达到：
  - `CVR=0.0%`
  - `feasibility_rate=100%`
  - `avg_vehicles_excess=0.0`
- 相对 PDF 表格：
  - 100-800 规模下 RL 平均总成本低于 PDF 表格；
  - 1000 规模下 RL 平均总成本略高于 PDF 表格；
  - RL 求解时间为秒级，显著低于 PDF 表格中的 OR-Tools 求解时间，但该时间只代表训练后推理与 decoder，不包含训练成本。
- `single_customer_cost` 随规模总体上升，主要原因是等待时间占比高；新口径下大规模等待占比约 `58%-62%`。
- `static100` 在多数规模下成本略低于 universal，说明当前 universal 的优势主要是单 checkpoint 多规模覆盖，而不是每个规模成本最优。

报告启示：
- 当前结果可支持“可行性优先 decoder 使 RL 在多规模静态 TWCVRP 上稳定达到 100% 可行率”。
- 成本层面应谨慎表述：100-800 规模具备明显优势，1000 规模略弱于 PDF 表格。
- 后续若继续优化，应优先降低等待时间，而不是继续只优化可行率。
