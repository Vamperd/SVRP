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

### 31. Hybrid 交通扰动测试后的再训练判断

用户完成 `benchmark_pdf_hybrid_100` 至 `benchmark_pdf_hybrid_1000` 的分规模交通扰动评估，并询问是否需要继续再训练。

阶段结论：
- 当前 hybrid 结果表明，静态训练模型在交通扰动下基本保持稳定：
  - `static100` 各规模平均成本相对静态只增加约 `0.018%-0.043%`；
  - `universal` 各规模平均成本相对静态只增加约 `0.019%-0.034%`；
  - 100/200/400/600 规模 corrected feasibility 均为 `1.0`；
  - 800/1000 规模仅出现极少量 Monte Carlo 样本级时间窗违约，`CVR` 接近 `0`。
- `benchmark_pdf_hybrid_*.csv` 中的 `feasibility_rate=0.0` 不能直接用于报告，这是 hybrid/traffic 聚合字段名不一致导致的统计显示问题；应以 JSON 中逐实例 `benchmark_feasibility` 重新聚合后的 corrected feasibility 为准。
- 在当前交通扰动强度和评估口径下，不建议立刻训练新的 traffic 模型；已有 static 模型已经能作为带扰动场景的有效鲁棒性结果。

报告启示：
- 可以将 hybrid 结果写为“静态训练策略在交通扰动下具有较强稳定性，平均成本波动极小，可行性基本保持”。
- 若后续需要更强对照，可以先修复 CSV 聚合字段，再考虑更强扰动或专门 traffic 训练；但当前阶段不应把时间投入大规模再训练。

### 32. 修复 hybrid/traffic CSV 可行率显示并准备强扰动对照

用户要求先修复 `benchmark_pdf_hybrid_*.csv` 中 `feasibility_rate=0.0` 的聚合显示问题，并说明后续希望尝试更强交通扰动，以比较 `static_rl` 与 `traffic_rl` 的训练区别。

阶段结论：
- 已修复 `benchmark_evaluate.py` 的聚合逻辑：
  - 静态单样本行继续兼容 `benchmark_feasible` 布尔字段；
  - hybrid/traffic 的 Monte Carlo 聚合行兼容 `benchmark_feasibility` 小数可行率字段；
  - CSV、JSON aggregate 后续都会使用一致的 `feasibility_rate`。
- 已通过 `python -m py_compile` 和聚合函数 sanity check，确认 `benchmark_feasibility=1.0/0.5` 可正确汇总为 `0.75`，不再被当作缺失字段归零。
- 后续若做“更强交通扰动”，需要注意当前 PDF 同口径评估中的 `--traffic_sigma` 不是主要扰动强度来源；更合理的下一步是增加评估侧扰动强度参数，再分别评估 `static_rl` 与 `traffic_rl`。

报告启示：
- 修复前的 hybrid CSV 可行率不能作为最终报告依据；修复后重新生成 CSV 才可直接入表。
- 强扰动实验应作为新的鲁棒性对照组，不应混入当前标准 PDF 同口径结果。

### 33. 生成 RL-TWCVRP 多规模实验 Markdown 报告

用户要求基于当前所有测试结果，仿照 `benchmark_report.pdf` 的大体格式生成 Markdown 报告，并重点说明模型组成、奖励函数、训练方法、泛化 detector 头设计、静态结果和交通扰动结果。

阶段结论：
- 已生成 `svrpbench/models/rl_solomon_tw/results/RL_TWCVRP_EXPERIMENT_REPORT.md`。
- 报告按 PDF 结构组织为：
  - 任务与评估口径；
  - 模型组成；
  - 训练方法与奖励函数；
  - 静态 TWCVRP 结果；
  - Hybrid 交通扰动鲁棒性；
  - 与 PDF 报告对照；
  - 综合结论与后续建议。
- 报告中将“泛化 detector 头”按当前代码解释为动态可行性特征头：它并非独立分类器，而是将候选客户的 `travel/arrival/wait/late/slack/capacity` 等动态约束风险特征编码后并入 pointer 打分。
- 报告结果明确区分：
  - `static100`：100 规模训练模型，在 100-1000 上做 zero-shot 泛化；
  - `universal`：100/200/400/600/800/1000 多规模训练模型；
  - 静态 PDF 同口径评估；
  - hybrid 交通扰动评估。

报告启示：
- 当前报告可作为阶段性实验总结，用于说明独立 RL 链路已经具备多规模可行求解能力。
- 报告中的后续建议应聚焦等待时间压缩和更强交通扰动下的 `static_rl`/`traffic_rl` 对照，而不是继续只优化可行率。

### 34. 新增强交通扰动评估参数 `--traffic_strength`

用户决定先推进强交通扰动评估，并要求先实现 `--traffic_strength`。

阶段结论：
- 已在 `benchmark_evaluate.py` 中新增 `--traffic_strength`，默认值为 `1.0`，保持此前 PDF 风格 baseline 评估兼容。
- 该参数只放大 stochastic traffic delay 和 accident delay，不放大基础欧氏距离：
  - `traffic_strength=0.0` 等价于无随机交通延迟；
  - `traffic_strength=1.0` 等价于此前标准 hybrid/traffic 扰动；
  - `traffic_strength=3.0/5.0` 可用于强扰动压力测试。
- CSV/JSON 输出中已增加 `traffic_strength` 字段，便于后续按扰动强度整理实验表格。
- 已通过 `python -m py_compile benchmark_evaluate.py` 和函数级 sanity check。

报告启示：
- 后续强扰动实验应先从 `traffic_strength=3.0` 开始；若 `CVR` 和可行率仍几乎不变，再提高到 `5.0`。
- 强扰动结果应作为新实验组单独呈现，不覆盖此前标准 PDF 同口径 hybrid 结果。

### 35. 强交通扰动结果分析与 traffic 模型训练判断

用户完成强交通扰动评估，并询问是否需要训练扰动模型。

阶段结论：
- 已读取 `strong_hybrid_s3_100/400/1000` 以及 `strong_hybrid_s6/s10/s15_1000`。
- `traffic_strength=3.0` 下：
  - 100 和 400 规模两个模型仍 `CVR=0`、可行率 `100%`；
  - 1000 规模 `static100` 出现轻微时间窗违约，`CVR≈0.0015%`、可行率约 `99.57%`；
  - 1000 规模 `universal` 仍 `CVR=0`、可行率 `100%`。
- 1000 规模压力曲线中，即使提高到 `traffic_strength=15.0`：
  - `static100` 约 `CVR=0.0073%`、可行率约 `98.97%`；
  - `universal` 约 `CVR=0.0022%`、可行率约 `99.74%`。
- 平均总成本相对静态增长仍很小，1000 规模在 `traffic_strength=15.0` 下仅约 `+0.21%`。

报告启示：
- 当前扰动设置已经显示 `universal` 比 `static100` 更鲁棒，但退化幅度仍偏小。
- 不建议立即投入 traffic 模型训练；应先让扰动评估能稳定制造更明显的中大规模退化，否则 traffic_rl 很难在报告中体现充分增益。
- 下一步更合理的是强化评估侧交通模型，例如使用距离比例型延迟或更高强度压力测试，再决定是否训练 traffic_rl。

### 36. 增加距离比例型强交通扰动 `--traffic_profile proportional`

用户要求继续进行更强扰动。此前 `--traffic_strength` 只放大额外 additive delay，实测即使 `traffic_strength=15`，1000 规模退化仍然很小。

阶段结论：
- 已在 `benchmark_evaluate.py` 中新增 `--traffic_profile`：
  - `additive`：默认旧口径，`travel_time = distance + strength * delay`；
  - `proportional`：新强扰动口径，`travel_time = distance * (1 + strength * delay_ratio) + strength * accident_delay`。
- CSV/JSON 已增加 `traffic_profile` 字段，便于报告区分标准扰动和强扰动。
- 函数级 sanity check 显示同一条距离为 100 的边：
  - `additive, strength=3` 得到约 `100.25`；
  - `proportional, strength=3` 得到约 `124.52`；
  因此 proportional 能显著提高中大规模路径的时间窗压力。
- 已通过 `python -m py_compile benchmark_evaluate.py`。

报告启示：
- 后续强扰动实验建议使用 `--traffic_profile proportional`，先从 `traffic_strength=1.0` 或 `2.0` 开始，再按退化程度提高。
- 与 PDF 标准扰动结果对比时，必须标明这是额外 stress-test 口径，不应替代标准 PDF 同口径结果。

### 37. proportional 强扰动结果后的 traffic_rl 训练判断

用户完成 `stress_prop_s1_400/800/1000` 评估，并询问当前是否应该训练交通扰动模型。

阶段结论：
- `traffic_profile=proportional, traffic_strength=1.0` 已经制造出明显中大规模退化：
  - 400 规模：`static100` 可行率约 `82.22%`、`CVR≈0.0454%`；`universal` 可行率约 `88.89%`、`CVR≈0.0324%`；
  - 800 规模：两模型可行率降至约 `45%-48%`，`CVR≈1.0%`；
  - 1000 规模：两模型可行率降至约 `36%`，`CVR≈1.42%-1.45%`。
- 与 additive 强扰动不同，proportional s1 已能稳定暴露大规模时间窗脆弱性，因此具备训练 `traffic_rl` 的实验必要性。
- 不建议继续增大到 s2/s3 后再训练；当前 s1 已足够强，继续加大会让问题过难，难以区分训练改进和整体崩溃。

报告启示：
- 可以将 proportional s1 作为“强交通压力测试”主场景。
- 下一步应训练 traffic-aware 模型，并在同一 `traffic_profile=proportional, traffic_strength=1.0`、同一 test split、同一 `mc_samples=30` 和 `traffic_seed=42` 下与 static_rl 对比。
- 训练前最好补齐 100/200/600 的 s1 评估，形成完整 100-1000 强扰动基线表。

### 38. 接入强交通扰动训练脚本参数

用户补齐 proportional 强扰动测试后，要求给出可训练强交通扰动模型的修改后脚本。

阶段结论：
- 已将强交通扰动参数接入训练侧：
  - `train.py` 新增 `--traffic_profile {additive,proportional}`；
  - `train.py` 新增 `--traffic_strength`；
  - `traffic.py` 的 `planning_matrix` 和 `sample_traffic_matrix` 支持 proportional 强扰动；
  - `benchmark_evaluate.py --mode traffic` 的规划矩阵也使用同样的 profile/strength；
  - `solve.py` 支持同样参数，便于单实例推理和绘图。
- proportional 训练矩阵采用距离比例型扰动近似强交通压力，使 traffic-aware decoder 在训练时能看到更保守的旅行时间。
- 已通过 `python -m py_compile traffic.py train.py benchmark_evaluate.py solve.py`，并通过 CLI sanity check 确认 `train.py --help` 已显示新参数。

报告启示：
- 后续 `traffic_rl` 建议以 `traffic_profile=proportional, traffic_strength=1.0` 作为主训练口径。
- 评估对比时，`static_rl` 应使用 `mode=hybrid`，`traffic_rl` 应使用 `mode=traffic`；二者不能混在同一个 `benchmark_evaluate.py --mode` 调用中直接比较。

### 39. traffic_prop_s1_universal_v3 训练效果评估

用户完成 `traffic_prop_s1_universal_v3_best.pt` 在 100-1000 各规模上的 `mode=traffic` 强扰动评估，并要求判断训练效果。

阶段结论：
- 本轮 traffic_rl 不是最终可采用的强扰动模型，只能作为“traffic-aware 训练初版/消融结果”。
- 相比 `stress_prop_s1` 的静态模型 hybrid 基线，traffic_rl 在大规模上提高了部分实例的完全可行率：
  - 800 规模可行率约从 `45%-48%` 提升到 `56.58%`；
  - 1000 规模可行率约从 `36.41%` 提升到 `53.68%`。
- 但代价非常明显：
  - 800 规模 CVR 从约 `1.0%` 升到 `5.05%`；
  - 1000 规模 CVR 从约 `1.42%-1.45%` 升到 `5.98%`；
  - 600/800/1000 平均总成本分别比 universal hybrid baseline 高约 `65.5%/74.4%/81.2%`。
- 主要原因是 traffic_rl 形成了过度保守的多路线策略：
  - 400 路线数从约 `37-38` 增至 `63.8`；
  - 600 从约 `57` 增至 `102.8`；
  - 800 从约 `77-78` 增至 `141.3`；
  - 1000 从约 `97-99` 增至 `183.1`。

报告启示：
- 可以写为：traffic-aware 训练证明了“增加交通缓冲可以提升部分大规模实例的完全可行率”，但当前 reward/decoder 组合导致路线过度分散，CVR 和成本恶化。
- 下一步不应继续直接加长同一训练，而应调整目标函数和训练口径：提高 CVR 逐客户惩罚、限制 route_count 膨胀、加入 PDF 强扰动评估口径的验证/选择机制。

### 40. 实现 v4 强扰动 `robust_cvr` 训练目标

用户要求按“强交通扰动训练目标重构计划”实现 v4 训练目标，使 traffic_rl 更贴近报告需求：优先降低 CVR，其次提高可行率，再控制路线数和成本。

阶段结论：
- 已新增 `--objective robust_cvr`，保留默认 `feasibility` 目标以兼容旧训练。
- `robust_cvr` reward 改为按客户数归一化：
  - 惩罚 `time_window_violations / num_customers`；
  - 惩罚 `capacity_violations / num_customers`；
  - 惩罚 `late_minutes / num_customers`；
  - 成本使用 `total_cost / num_customers`；
  - `feasible_bonus` 和 `infeasible_penalty` 也按客户数缩放。
- 已新增路线膨胀控制：
  - `--target_customers_per_route`；
  - `--route_overuse_penalty`；
  - `route_overuse = max(0, route_count - ceil(num_customers / target_customers_per_route))`。
- 已调整 best checkpoint 选择逻辑：
  - `robust_cvr` 下优先最小化 `val_cvr`；
  - 其次最大化 `val_feasibility`；
  - 再最小化 `val_route_overuse` 和 `val_total_cost`。
- 已新增轻量 robust validation：
  - `--robust_val_samples`；
  - validation 中同一路线用多个 traffic sample 评估，减少训练 proxy 与最终强扰动评估的偏差。
- 日志和 history 增加：
  - `train/val_route_count`；
  - `train/val_route_overuse`；
  - `train/val_cost_per_customer`；
  - `train/val_late_per_customer`。
- 已通过：
  - `python -m py_compile train.py decoder.py traffic.py benchmark_evaluate.py`；
  - CLI 参数检查；
  - 单实例函数级 sanity check。

报告启示：
- v4 的重点不是继续追求“更多路线换完全可行率”，而是压低强扰动下的平均 CVR，同时避免路线数膨胀。
- 后续评估 v4 时，应重点对比 `avg_cvr`、`feasibility_rate`、`avg_route_count` 和 `avg_cost` 四个指标。

### 41. v4 smoke_b 训练结果判断

用户运行了 `traffic_prop_s1_v4_smoke_b`，该命令使用 `objective=robust_cvr`、`traffic_profile=proportional`、`traffic_strength=1.0`，训练规模为 `400/600/800/1000`。

阶段结论：
- 训练链路是正常的，日志中已出现 `val_cvr`、`val_route_overuse`、`val_cost_per_customer` 等 v4 诊断指标。
- `best_checkpoint` 被选在 epoch 1，这是符合当前 best 规则的：epoch 1 的 `val_cvr=5.676`，低于 epoch 5/10/15。
- 后续 epoch 虽然在部分指标上有改善，例如 epoch 10 的 `val_total_cost=126908.97`、`val_route_overuse=23.42`、`val_cost_per_customer=163.60`，但 `val_cvr` 升至 `6.064`，因此不能作为强扰动目标下的更优 checkpoint。
- 该 smoke 结果说明当前 v4 配置尚未真正降低验证集 CVR；继续简单拉长同一配置训练，收益不确定。

报告启示：
- v4 smoke_b 可以作为“强扰动目标重构后的中间实验”记录：它验证了 robust_cvr 指标链路可用，但也暴露出训练目标仍没有有效压低验证 CVR。
- 下一步不应只增加 epoch，而应优先调整目标权重、训练尺度或让验证口径更贴近最终 `benchmark_evaluate.py` 的强扰动 Monte Carlo 结果。

### 42. 转向 vrp_benchmark 外部数据可行性验证

用户表示当前不希望继续投入过多交通扰动训练尝试，希望改用 `vrp_benchmark` 数据验证当前模型的可行性。

阶段结论：
- 当前最稳妥路线是暂时不再使用旧的 `python -m vrp_bench solve` 接入口，而是继续使用独立 PyTorch 链路。
- 对 `vrp_benchmark/real_twcvrp/*.npz`：
  - `rl_solomon_tw/dataset.py` 已支持 `.npz` 读取；
  - `solve.py --input *.npz` 可用于单文件推理；
  - `evaluate.py --data_root ... --source npz` 或 `benchmark_evaluate.py --eval_files ...` 可用于批量评估。
- 对 `vrp_benchmark/real_cvrp/*.npz`：
  - 应使用 `rl_standalone` 的 CVRP 链路；
  - 该链路适合验证“CVRP 数据读取 -> RL 推理 -> 路线 JSON 输出”的闭环；
  - checkpoint 仍按客户规模绑定，跨规模需要重新训练或改造模型。
- 推荐优先验证 single-depot 数据：
  - TWCVRP：`real_twcvrp/twvrp_{size}_single_depot.npz`；
  - CVRP：`real_cvrp/cvrp_{size}_single_depot_single_vehicule_sumDemands.npz`。

报告启示：
- `vrp_benchmark` 外部验证可以作为 Solomon/Homberger 实验之后的“跨数据源可行性验证”。
- 首要指标仍应是 `feasibility`、`cvr`、`vehicles_excess`、`missing_customers` 和 `duplicate_visits`，成本指标放在第二层解释。

### 43. static100 在 vrp_benchmark TWCVRP100 上的首次外部验证

用户使用 `static_100_v2_full_best.pt` 对 `vrp_benchmark/real_twcvrp/twvrp_100_single_depot.npz` 进行静态推理，输出 `results/vrpbench_tw100_static100_solution.json`。

阶段结论：
- 10 个实例中 `4/10` 完全可行，平均 `cvr=0.90`。
- 所有实例均无漏访、无重复访问、无容量违约、无车辆数超限。
- 不可行实例全部由时间窗迟到造成：
  - 平均 `time_window_violations=0.9`；
  - 总计 9 个时间窗违约；
  - 平均 `late_minutes=37.2`。
- 平均路线数约 `23.7`，车辆约束保持稳定，说明 `strict_insert` 对车辆数和访问完整性的结构性控制仍然有效。

报告启示：
- 这是一个典型跨数据源验证结果：模型没有崩溃，仍能服务全部客户并满足容量/车辆约束，但时间窗泛化明显弱于 Solomon/Homberger 测试集。
- 报告中可表述为“Solomon 训练模型在 vrp_benchmark TWCVRP100 上具备路线生成与基本约束保持能力，但时间窗分布迁移导致完全可行率下降”。
- 下一步应优先比较 `tw_universal_all_full_best.pt` 在同一数据上的结果，而不是立即继续交通扰动训练。

### 44. 实现 vrp_benchmark TW100 fine-tune 接入

用户要求按计划实现以 `vrp_benchmark/real_twcvrp/twvrp_100_single_depot.npz` 为目标数据集的 fine-tune 与 native 评估入口。

阶段结论：
- 已在 `train.py` 新增 `--init_checkpoint`，可从 `static_100_v2_full_best.pt` 等已有 checkpoint 初始化后继续训练。
- checkpoint 元数据现在会额外记录 `split_indices`，便于报告中明确 train/val/test 实例编号。
- 已在 `benchmark_evaluate.py` 新增 `--metric_profile native|pdf_compatible`：
  - 默认 `pdf_compatible` 保持旧报告口径；
  - `native` 使用项目 evaluator 与 `vrp_benchmark` 自带 `time_matrix`，作为 fine-tune 的主验收口径。
- 修复了 `.npz` 规模匹配问题：`--source npz --size 100` 不再误匹配 `twvrp_1000_single_depot.npz`。
- 已通过：
  - `python -m py_compile train.py benchmark_evaluate.py dataset.py decoder.py evaluator.py`；
  - `train.py --help` 中确认 `--init_checkpoint`；
  - `benchmark_evaluate.py --help` 中确认 `--metric_profile`；
  - 临时 native 评估 smoke，test split 输出 `base` 指标：`instances=2`、`avg_cvr=1.0`、`feasibility_rate=0.5`。
- 0 epoch 初始化检查确认固定 split 索引为：
  - train: `[2, 3, 4, 5, 7, 9]`
  - val: `[6, 8]`
  - test: `[0, 1]`

报告启示：
- 现在可以正式进行 `vrp_benchmark` TW100 留出 fine-tune 实验。
- 若 fine-tune 后 test split 的 `avg_cvr` 和 `late_minutes` 不能下降，可将结论推进到“当前 policy 排序 + strict_insert 对难时间窗分布仍不足，需要 time-window repair 或 deadline-aware decoder”。

### 45. vrp_benchmark TW100 smoke fine-tune 结果判断

用户完成 `vrpbench_tw100_static100_ft_smoke_best.pt` 在留出 test split 上的 native 口径评估，并询问是否值得继续正式 fine-tune。

阶段结论：
- smoke fine-tune 没有改善 test split 结果；`base` 与 `smoke` 指标完全一致：
  - `avg_cost=31305.5`
  - `single_customer_cost=313.055`
  - `avg_waiting=7706.0`
  - `avg_cvr=1.0`
  - `feasibility_rate=0.5`
  - `avg_route_count=23.5`
- per-instance 也完全一致：
  - test instance 0 可行，`cvr=0`；
  - test instance 1 不可行，`time_window_violations=2`、`late_minutes=51.0`。
- smoke checkpoint 的 best epoch 停在 epoch 1，validation 仍为 `val_feasibility=0.0`、`val_cvr=2.0`，说明当前短 fine-tune 没有让 greedy 策略发生有效改变。

执行建议：
- 不建议直接运行原计划的 80 epoch 正式 fine-tune 命令，因为当前 smoke 没有显示正向信号。
- 若仍要验证“继续 RL 微调是否可能修复 TW100”，应先改为更强但较短的 v2 fine-tune：提高学习率和时间窗惩罚、取消路线数惩罚、缩短到 40 epoch，并严格看 validation/test 的 `avg_cvr` 与 `late_minutes` 是否下降。
- 如果 v2 仍无改善，下一步应停止单纯 fine-tune，转向 `time_window_repair` 或 `deadline-aware decoder`，因为问题更可能在解码/修复结构，而不是训练轮次不足。

### 46. 实现 time-window repair 与 deadline-aware decoder，并修正 TW100 结论

用户要求实现“时间窗优先 decoder/repair”计划，用于验证当前策略是否能更好适应 `vrp_benchmark TW100` 的窄时间窗分布。

阶段实现：
- 在 `decoder.py` 中新增 `deadline_aware_insert`：
  - 不再只按旅行增量筛选插入位置；
  - 候选位置按时间窗违约数、迟到分钟、路线数、最小 slack、等待增量、成本增量排序；
  - 保留 `strict_insert` 作为 baseline。
- 在 `decoder.py` 中新增 `post_opt=time_window_repair`：
  - 围绕迟到客户及其前后邻居做 relocate；
  - 再尝试与其他客户 swap；
  - 只接受不破坏访问完整性、容量、车辆数，且指标字典序更优的移动。
- 在 `solve.py`、`benchmark_evaluate.py`、`evaluate.py`、`train.py` 中接入：
  - `--decoder deadline_aware_insert`
  - `--post_opt none|time_window_repair`
- 已通过 `python -m py_compile decoder.py solve.py benchmark_evaluate.py train.py evaluate.py`。

关键验证：
- 对 `static_100_v2_full_best.pt` 在 `vrp_benchmark TW100` 全 10 实例上做临时 native 评估：
  - `strict_insert + time_window_repair`：`avg_cvr=0.9`、`feasibility_rate=0.4`，成本从约 `31290.7` 小幅降到约 `31272.4`；
  - `deadline_aware_insert + time_window_repair`：`avg_cvr=0.9`、`feasibility_rate=0.4`，成本更高，暂不优于 strict baseline。
- 进一步检查发现，`vrp_benchmark TW100` 的 9 个迟到违约刚好对应 9 个“从 depot 直接出发也超过 due time”的客户：
  - instance 1: 2 个；
  - instance 2: 1 个；
  - instance 3: 1 个；
  - instance 6: 2 个；
  - instance 8: 2 个；
  - instance 9: 1 个。
- 这意味着在当前 native `time_matrix` 口径下，这些客户存在静态时间窗不可达下界；因此 `4/10 feasible, avg_cvr=0.9` 很可能不是模型/decoder 单独造成的失败，而是数据口径本身包含不可完全满足的时间窗。

报告启示：
- 不能简单写成“static_insert 泛化失败”；更准确的说法是：
  - 当前链路已经达到 `vrp_benchmark TW100` native 口径下由 depot 直达时间窗下界所限制的可行性水平；
  - `strict_insert` 仍有效保证无漏访、无重复、容量可行和车辆数不超限；
  - 对这批数据若要追求 100% 可行率，需要调整时间窗口径、允许软时间窗、或验证 benchmark 原始时间矩阵/时间窗是否设计为硬可行。
- 后续改进方向应从“强行 fine-tune”转向：
  - 在评估中显式报告 `direct-depot infeasible lower bound`；
  - 对硬不可达客户采用软时间窗迟到惩罚；
  - 或引入允许等待/出发时间偏移/数据修复的业务假设。

### 47. 交通扰动时间尺度归一化假设与 A/B 验证入口

用户提出：交通扰动实验失效可能来自 Solomon/Homberger 与 SVRP 的时间尺度不一致。SVRP 的交通扰动函数使用固定日内分钟峰值，例如早高峰 `480`、晚高峰 `1020`、事故峰 `1260`；而 Solomon/Homberger 的 depot due time 从数百到数千不等，直接使用原始 `current_time` 可能让扰动落在错误时间段。

阶段判断：
- 该猜测方向正确，但需要区分训练侧和评估侧：
  - 当前 `traffic.py` 的训练侧扰动主要是整张时间矩阵乘因子，不使用 route traversal 的实时 `current_time`；
  - `benchmark_evaluate.py` 的 PDF/SVRP 风格 hybrid/traffic 评估会逐边使用 `current_time`，因此时间尺度错位主要影响评估侧 PDF traffic 口径。
- 在验证时间尺度假设前，应先把强扰动参数回退到旧/标准扰动强度：
  - `traffic_profile=additive`
  - `traffic_strength=1.0`
  - `traffic_sigma=0.2`
  - `traffic_buffer=0.5`
  - `decoder=strict_insert`
  - `post_opt=none`
- 不建议删除强扰动代码或 checkpoint，只通过参数做非破坏式实验回退。

阶段实现：
- 在 `benchmark_evaluate.py` 新增 `--traffic_time_scale raw|depot_day`：
  - `raw`：旧口径，直接使用原始 `current_time`；
  - `depot_day`：将 depot 时间窗 `[ready_0, due_0]` 映射到 `[0, 1440]` 后再计算 SVRP 早晚高峰和事故扰动。
- 新增诊断字段并写入 CSV/JSON：
  - `traffic_time_scale`
  - `avg_depot_due`
  - `avg_traffic_edges`
  - `avg_raw_current_time`
  - `avg_scaled_current_time`
  - `avg_delay`
  - `avg_delay_ratio`
- 更新 `README_TW_RL.md`，加入 raw/depot_day A/B 验证命令。
- 已通过：
  - `python -m py_compile traffic.py benchmark_evaluate.py train.py evaluate.py solve.py`
  - 100 规模 `mc_samples=2` 临时 raw/depot_day A/B sanity check。

初步 sanity 结果：
- 在 `universal_v1/test/100` 上，raw 与 depot_day 的差异很小：
  - raw: `avg_raw_current_time≈402.38`，`avg_scaled_current_time≈402.38`，`avg_delay_ratio≈0.00231`；
  - depot_day: `avg_raw_current_time≈402.38`，`avg_scaled_current_time≈419.75`，`avg_delay_ratio≈0.00232`。
- 这说明 100 规模不能充分验证该假设；应重点在 `600/800/1000` 等 depot horizon 明显偏离 1440 的规模上做 A/B。

报告启示：
- 后续是否训练 traffic_rl，应先看 raw/depot_day 在大规模上的 `avg_delay_ratio`、`avg_cvr`、`feasibility_rate` 是否显著不同。
- 若差异明显，可说明此前交通扰动对 Solomon/Homberger 时间尺度不够敏感；若差异很小，则问题更可能来自训练目标和路线数膨胀，而不是时间归一化。

### 49. 新建 Event-Driven Online Recourse RL 实验线

用户提出在线追溯式 RL 架构优化建议：采用事件驱动环境，避免 `车辆数 x 客户数` 动作空间爆炸；引入时间编码；使用 hard mask 避免 RL 早期崩溃。

阶段实现：
- 新建独立实验目录 `svrpbench/models/rl_recourse_tw/`，不替换当前 `rl_solomon_tw` 主链路。
- 实现 `EventDrivenTWEnv`：
  - 每步选择最早空闲车辆；
  - 策略只为该车选择下一个客户；
  - hard mask 屏蔽已服务、容量超限、最快到达也必迟到客户；
  - 若全部时间窗 mask，则 fallback 到容量可行客户并记录 `forced_late_actions`。
- 实现 `EventDrivenSTPolicy`：
  - 当前车辆作为 query；
  - 客户节点作为 key/value；
  - 客户特征中包含坐标、需求、时间窗、预计到达、等待、迟到、slack、sin/cos 时间编码和 legal mask。
- 实现 `rollout.py`、`heuristic.py`、`train.py`、`evaluate.py`：
  - 支持 `earliest_due`、`min_late`、`recourse`、`strict_insert` 对比；
  - 支持 `risk_objective=mean|cvar`；
  - checkpoint 保存普通字典，不保存环境对象。
- 新增 `README_RECOURSE_RL.md`，提供 smoke 训练和评估命令。

验证结果：
- 已通过：
  - `python -m py_compile common.py env.py policy.py rollout.py heuristic.py train.py evaluate.py`
  - 100 规模静态启发式临时评估；
  - 1 epoch/1 step smoke 训练；
  - smoke checkpoint 加载评估。
- 临时 100 静态测试结果：
  - `earliest_due`：`feasibility≈0.778`、`avg_cvr≈0.667`、`late_minutes≈8.22`；
  - 未充分训练的 `recourse`：`feasibility≈0.222`、`avg_cvr≈9.0`、`late_minutes≈2875.6`。

报告启示：
- 事件驱动环境本身是可用的，且简单在线启发式已经显示比随机未训练策略稳定得多。
- 当前 recourse 神经策略仍需正式训练或 imitation warm-start；不能用 smoke checkpoint 的低性能评价该架构。
- 后续优先比较 `earliest_due/min_late` 与 `strict_insert` 在交通扰动下的表现，再决定是否投入更长 RL 训练。

### 48. 交通时间尺度 A/B：400 与 600 规模初步结果

用户完成 `traffic_time_scale=raw` 与 `traffic_time_scale=depot_day` 在 400、600 规模上的标准扰动 A/B 评估，并要求比较结果。

阶段结论：
- `depot_day` 确实改变了交通函数看到的时间坐标：
  - 400 规模中，`avg_scaled_current_time` 从约 `633/628` 降到约 `398/393`；
  - 600 规模中，`avg_scaled_current_time` 从约 `794/783` 降到约 `385/381`。
- 但该变化没有显著改变扰动强度和最终结果：
  - 400 规模：
    - static100 `avg_delay_ratio` 从 `0.002130` 到 `0.002133`，`avg_cvr=0.0`、`feasibility=1.0` 不变；
    - universal `avg_delay_ratio` 从 `0.002125` 到 `0.002128`，`avg_cvr=0.0`、`feasibility=1.0` 不变。
  - 600 规模：
    - static100 `avg_delay_ratio` 从 `0.002106` 到 `0.002015`，`avg_cvr=0.0`、`feasibility=1.0` 不变；
    - universal `avg_delay_ratio` 从 `0.002018` 到 `0.002009`，`avg_cvr≈0.0598`、`feasibility≈0.9744` 不变。
- 成本、等待、robustness 标准差几乎没有变化，说明在标准扰动强度下，400/600 的 raw vs depot_day 时间尺度不是主要影响因素。

报告启示：
- 时间尺度归一化是合理假设，但 400/600 结果暂不支持它是当前交通扰动实验失效的主因。
- 当前 PDF/SVRP 交通函数的 `time_factor` 峰值幅度本身很小，且 additive 扰动的平均 delay ratio 约只有 `0.2%`，因此即使时间坐标变化，最终路径指标也几乎不动。
- 下一步可继续看 800/1000，但若仍无明显差异，后续重点应转向交通扰动强度公式、扰动注入方式和 traffic-aware 训练目标，而不是单纯时间归一化。
### 50. Event-driven recourse 实验阶段性结论与后续方向

用户完成了四组 recourse 实验：启发式与 `strict_insert` 基线对比、recourse smoke 训练、100 规模正式训练，以及 100 模型跨 200/400 规模评估。该节点用于判断“事件驱动在线追溯”方向是否值得继续。

阶段性结果：
- `earliest_due` 启发式在可行性和 CVR 上明显优于 `strict_insert`，尤其在 200/400 规模上接近或达到完全可行：
  - 200 规模：`earliest_due feasibility≈0.9667`、`avg_cvr≈0.0167`，而 `strict_insert feasibility≈0.2889`、`avg_cvr≈1.1722`。
  - 400 规模：`earliest_due feasibility≈0.9963`、`avg_cvr≈0.0009`，而 `strict_insert feasibility≈0.1074`、`avg_cvr≈1.2417`。
- 这说明事件驱动环境、异步车辆决策和 hard mask 对时间窗可行性非常有效。
- 但当前训练得到的神经 `recourse` 策略尚未成功：
  - 100 规模测试：`recourse feasibility≈0.5111`，略高于 `strict_insert≈0.4926`，但 `avg_cvr≈7.1778`、`late_minutes≈898`，显著差于 `strict_insert` 和 `earliest_due`。
  - 200/400 跨规模测试中，`recourse` 的可行率高于 `strict_insert`，但 CVR 与迟到分钟数明显更差，说明它更多是在“用满车辆数”而不是学到稳定的时间窗排序策略。
- 100 规模训练日志显示 best checkpoint 停在较早 epoch，validation 的 `val_cvr` 和 `val_feasibility` 没有形成稳定改善，说明从零开始 REINFORCE 方差过大，当前奖励不足以让策略靠探索学到 `earliest_due` 这类强启发式行为。

执行判断：
- 不建议继续直接加长当前 recourse RL 训练，也不建议立刻做 600/800/1000 的神经 recourse 长训。
- 应先把 `earliest_due` 作为专家策略，加入 imitation warm-start / behavior cloning，让模型先学会低 CVR、低迟到的动作分布，再进入 RL 微调。
- 下一阶段推荐对比：
  - `strict_insert`：低成本但时间窗可行性弱；
  - `earliest_due`：高可行性但路线数和成本高；
  - `imitation-only recourse`：验证神经策略能否复刻启发式；
  - `imitation + RL recourse`：在保持可行性的基础上降低成本和路线数。

报告表述建议：
- 可以写“事件驱动 recourse 框架被启发式结果验证为有效，尤其在时间窗可行性上显著优于一次性静态排序解码；但当前纯 REINFORCE 神经策略尚未达到启发式水平，需要引入专家模仿和分阶段训练。”
### 51. Recourse imitation warm-start 实现

基于前一阶段结论，用户要求实现 `recourse` 模型的专家模仿预训练，避免继续从零开始依赖高方差 REINFORCE 探索。

阶段实现：
- 在 `svrpbench/models/rl_recourse_tw/train.py` 中新增 imitation 参数：
  - `--imitation_epochs`
  - `--expert_strategy earliest_due|min_late`
  - `--imitation_weight`
  - `--bc_weight_after_imitation`
- 新增专家监督 rollout：
  - 每个 event-driven 决策状态下由专家策略选择客户；
  - 策略网络对专家动作计算 cross entropy；
  - 环境执行专家动作并记录专家路线指标；
  - 日志新增 `phase`、`imit_loss`、`bc_loss`、`train_imit_accuracy`。
- REINFORCE 阶段保留原有逻辑，同时可加入少量 BC 正则，防止策略在 RL 微调时偏离低 CVR 专家行为。
- 更新 `README_RECOURSE_RL.md` 为中文说明，加入 imitation smoke、正式训练、100 测试和跨规模评估命令。

执行判断：
- 当前推荐先运行 100 规模 imitation smoke，观察 `train_acc`、`train_cvr`、`val_cvr` 是否明显优于旧 recourse。
- 若 `recourse_100_imitation_best.pt` 的 100 测试 `avg_cvr` 仍明显差于 `strict_insert`，应暂停多规模长训，继续改专家策略或 reward。
### 52. recourse_100_bc_long 跨 600/800 规模评估

用户完成 `recourse_100_bc_long_best.pt` 在 600 与 800 规模上的标准弱交通扰动评估，并要求判断后续方向。当前结果目录中未发现 1000 规模输出文件，因此本节点先基于 600/800 结论。

阶段结果：
- 600 规模：
  - `recourse avg_cvr≈0.0853`、`feasibility≈0.8598`，明显优于 `strict_insert avg_cvr≈1.3473`、`feasibility≈0.10`。
  - 但 `recourse` 弱于 `earliest_due avg_cvr≈0.0184`、`feasibility≈0.9239`，说明 100 规模训练模型在 600 上仍有泛化，但已经开始落后专家。
  - `recourse avg_cost≈256383`，低于 `earliest_due≈265774`，高于 `strict_insert≈73338`。
- 800 规模：
  - `recourse avg_cvr≈0.4223`、`feasibility≈0.7667`，仍明显优于 `strict_insert avg_cvr≈1.9932`、`feasibility≈0.0744`。
  - 但已经明显弱于 `earliest_due avg_cvr≈0.0474`、`feasibility≈0.8590`。
  - `recourse late_minutes≈2052`，高于 `earliest_due≈285` 与 `strict_insert≈207`，说明大规模下迟到严重度开始上升。

执行判断：
- `recourse_100_bc_long_best.pt` 已证明 100 训练模型具备跨 600/800 的时间窗鲁棒性泛化，仍显著优于静态 `strict_insert`。
- 但从 600 到 800，模型与 `earliest_due` 的差距明显扩大，说明不应直接把 100 模型作为最终大规模模型。
- 下一步应先补跑 1000 评估；若 1000 继续退化，应训练 100/200/400/600/800 的多规模 imitation+RL 模型，再考虑路线数/成本压缩。

### 53. recourse_100_bc_long 跨 1000 规模评估与下一阶段判断

用户补跑 `recourse_100_bc_long_best.pt` 在 1000 规模上的标准弱交通扰动评估。

阶段结果：
- 1000 规模：
  - `earliest_due avg_cvr≈0.0403`、`feasibility≈0.8453`、`late_minutes≈491.49`、`avg_cost≈656782`。
  - `recourse avg_cvr≈0.2674`、`feasibility≈0.7359`、`late_minutes≈2146.72`、`avg_cost≈637539`。
  - `strict_insert avg_cvr≈2.4697`、`feasibility≈0.0444`、`late_minutes≈429.95`、`avg_cost≈199793`。
- `recourse` 在 1000 上仍显著优于 `strict_insert` 的 CVR 与完全可行率，说明 100 规模训练出的 online recourse 策略具有真实跨规模泛化能力。
- 但相对 `earliest_due`，`recourse` 在 800/1000 上差距明显扩大，尤其迟到分钟数偏高，说明单 100 规模训练不足以作为最终大规模模型。

执行判断：
- 当前 `recourse_100_bc_long_best.pt` 可作为报告中的“100 训练模型跨规模泛化”核心结果。
- 后续若继续优化，应优先训练覆盖大规模的 recourse 模型，例如 `400/600/800/1000` 或 `100/200/400/600/800/1000` 多规模 imitation+RL。
- 不建议立刻做成本/路线数压缩，因为大规模时间窗鲁棒性还没有追平专家；应先提升 800/1000 的 CVR 和 feasibility。
