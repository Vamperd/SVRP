# Solomon/TWCVRP 强化学习当前链路说明

本文档只保留当前正在使用的独立 PyTorch TWCVRP 强化学习链路。旧的 RL4CO/TorchRL/Lightning 接入、旧 `greedy_split` 主流程和早期固定规模实验命令不再作为主线使用。

当前策略目标不是先追求最低成本，而是先得到可行路线，再逐步降低距离、等待时间和车辆数。

## 1. 当前默认策略

当前主线为 v2 可行性优先策略：

- 模型：`TWPointerPolicy`，共享节点编码器 + attention/pointer 解码。
- 解码器：默认使用 `strict_insert`。
- 约束优先级：车辆数、容量、时间窗可行性优先，然后再优化路线成本。
- 训练方式：先使用启发式 imitation warm-start，再进入 REINFORCE。
- checkpoint 选择：优先 `val_feasibility` 最大，其次 `val_cvr` 最小，最后 `val_total_cost` 最小。
- 加速参数：使用 `--insert_top_k`、`--val_limit`、较小 `steps_per_epoch` 控制训练耗时。

已经验证的方向：

```text
results/static_100_v2_smoke_test.json

static_rl:
  feasibility = 1.0
  cvr = 0.0
  vehicles_excess = 0.0

hybrid_rl:
  主要失败来源转为交通扰动下 time_window_violations
```

因此后续不应回退到旧 decoder，而应在 `strict_insert` 基础上继续优化速度和交通扰动鲁棒性。

## 2. 环境检查

所有命令都由你在 conda `svrp` 环境中手动执行：

```powershell
conda activate svrp
cd C:\Users\86136\Desktop\code\RL\SVRP\svrpbench\models\rl_solomon_tw
python -c "import torch, numpy; print(torch.__version__, torch.cuda.is_available())"
```

如果输出为 `False`，说明当前环境没有使用 CUDA 版 PyTorch，训练会明显变慢。此文档不自动安装环境。

## 3. 数据来源

当前 TWCVRP 主线使用本地 Solomon/Homberger 文本数据：

```text
C:\Users\86136\Desktop\code\RL\SVRP\solomon
```

先检查可识别的数据规模：

```powershell
python inspect_data.py --data_root C:\Users\86136\Desktop\code\RL\SVRP\solomon --sizes 100 200 400 600 800 1000
```

默认文件级划分：

- `train_ratio=0.70`
- `val_ratio=0.15`
- 剩余为 test
- `seed=1234`
- 按 `C1/C2/R1/R2/RC1/RC2` 分层划分

## 4. 静态 100 规模快速训练

如果 100 规模训练已经很慢，优先使用下面的快速确认命令。它会牺牲一部分训练充分性，但可以快速验证流程和趋势：

```powershell
python train.py --data_root C:\Users\86136\Desktop\code\RL\SVRP\solomon --size 100 --mode static --decoder strict_insert --insert_top_k 10 --imitation_epochs 1 --epochs 30 --steps_per_epoch 3 --batch_size 2 --val_every 5 --val_limit 4 --checkpoint checkpoints\static_100_v2_quick.pt
```

如果仍然过慢，继续降低：

```text
--batch_size 1
--steps_per_epoch 1
--val_limit 2
```

不建议一开始把 `--insert_top_k` 降到 `1`，路线质量容易明显变差。建议先用 `10`，稳定后再尝试 `30`。

## 5. 静态测试集评估

训练完成后，用 best checkpoint 在 test split 上评估：

```powershell
python evaluate.py --data_root C:\Users\86136\Desktop\code\RL\SVRP\solomon --sizes 100 --split test --decoder strict_insert --insert_top_k 10 --static_checkpoint checkpoints\static_100_v2_quick_best.pt --mc_samples 5 --output_json results\static_100_v2_quick.json --output_csv results\static_100_v2_quick.csv
```

静态阶段优先看：

- `static_rl.feasibility`
- `static_rl.cvr`
- `vehicles_excess`
- `time_window_violations`
- `total_cost`
- `route_count`

当前验收目标：

```text
static_rl feasibility = 1.0
static_rl cvr = 0.0
vehicles_excess = 0.0
```

在这些指标稳定后，再比较 `total_cost`、`waiting_time`、`route_count`。

## 6. 交通扰动训练

Solomon 原始文件不包含交通扰动矩阵。当前脚本会按固定 seed 在线生成扰动，不需要提前生成缓存。

交通扰动快速训练命令：

```powershell
python train.py --data_root C:\Users\86136\Desktop\code\RL\SVRP\solomon --size 100 --mode traffic --decoder strict_insert --insert_top_k 10 --traffic_sigma 0.2 --traffic_buffer 0.8 --imitation_epochs 1 --epochs 30 --steps_per_epoch 3 --batch_size 2 --val_every 5 --val_limit 4 --checkpoint checkpoints\traffic_100_v2_quick.pt
```

其中：

- `--traffic_sigma 0.2` 控制扰动强度。
- `--traffic_buffer 0.8` 给时间窗预留更保守的缓冲。
- `mode=traffic` 训练得到的是 traffic-aware 模型。

只有在需要审计或固定保存扰动矩阵时，才使用 `prepare_traffic.py`。正常训练和评估不需要这一步。

## 7. 静态/交通对照评估

使用静态模型和交通模型做对照：

```powershell
python evaluate.py --data_root C:\Users\86136\Desktop\code\RL\SVRP\solomon --sizes 100 --split test --decoder strict_insert --insert_top_k 10 --static_checkpoint checkpoints\static_100_v2_quick_best.pt --traffic_checkpoint checkpoints\traffic_100_v2_quick_best.pt --mc_samples 30 --traffic_sigma 0.2 --output_json results\v2_static_traffic_100.json --output_csv results\v2_static_traffic_100.csv
```

评估中三组含义：

- `static_rl`：静态模型在静态环境下评估。
- `hybrid_rl`：静态模型在交通扰动下评估。
- `traffic_rl`：交通扰动训练模型在交通扰动下评估。

交通扰动阶段优先目标：

```text
traffic_rl cvr < hybrid_rl cvr
traffic_rl time_window_violations < hybrid_rl time_window_violations
vehicles_excess = 0.0
```

## 8. 单实例推理与 SVG 绘图

使用训练好的 checkpoint 求解单个 Solomon 文件：

```powershell
python solve.py --input C:\Users\86136\Desktop\code\RL\SVRP\solomon\100\c101.txt --checkpoint checkpoints\static_100_v2_quick_best.pt --mode static --decoder strict_insert --insert_top_k 10 --output results\c101_static_solution.json --plot_svg results\plots\c101_static.svg
```

SVG 绘图不依赖 Matplotlib，通常可以避免 Windows 上 OpenMP DLL 冲突。

## 9. 多规模通用模型

如果希望一个 checkpoint 同时处理 `100/200/400/600/800/1000`，使用文件级数据划分：

```powershell
python create_universal_split.py --source_root C:\Users\86136\Desktop\code\RL\SVRP\solomon --output_root data_splits\universal_v1
python inspect_split.py --split_root data_splits\universal_v1
```

预期数量：

```text
100:  39 train / 8 val / 9 test
200:  42 train / 9 val / 9 test
400:  42 train / 9 val / 9 test
600:  12 train / 9 val / 39 test
800:  12 train / 9 val / 39 test
1000: 12 train / 9 val / 39 test
```

多规模快速训练先从小配置开始：

```powershell
python train.py --split_root data_splits\universal_v1 --sizes 100 200 400 600 800 1000 --mode static --decoder strict_insert --insert_top_k 10 --imitation_epochs 1 --epochs 10 --steps_per_epoch 2 --batch_size 1 --val_every 5 --val_limit 2 --checkpoint checkpoints\tw_universal_static_quick.pt
```

多规模测试：

```powershell
python evaluate.py --split_root data_splits\universal_v1 --sizes 100 200 400 600 800 1000 --split test --decoder strict_insert --insert_top_k 10 --static_checkpoint checkpoints\tw_universal_static_quick_best.pt --mc_samples 5 --output_json results\universal_static_quick.json --output_csv results\universal_static_quick.csv
```

注意：`400/600/800/1000` 在 `strict_insert` 下会显著变慢。先让 100 规模稳定，再推进多规模训练。

## 10. 速度控制建议

训练慢的主要瓶颈在 Python/CPU 侧的 `strict_insert` 插入评估，而不是神经网络前向本身。

优先按这个顺序加速：

1. 降低 `--steps_per_epoch`。
2. 降低 `--batch_size`。
3. 使用 `--val_limit` 减少训练期间验证开销。
4. 使用 `--insert_top_k 10` 做候选插入剪枝。
5. 增大 `--val_every`，减少验证频率。
6. 确认 CUDA 环境启用，但不要期待 CUDA 解决全部慢的问题。

建议不要为了速度回退到旧 `greedy_split`，因为旧策略的主要问题是车辆数超限，容易让可行率失真。

## 11. 输出文件与指标解释

常见输出：

- `checkpoints\*_best.pt`：验证集最优 checkpoint。
- `results\*.json`：完整评估结果，适合报告复盘。
- `results\*.csv`：聚合表格，适合复制进表格或画图。
- `results\plots\*.svg`：单实例路线图。

关键指标：

- `feasibility`：可行率，当前最优先。
- `cvr`：constraint violation rate，越低越好。
- `vehicles_excess`：超出允许车辆数的平均数量，应为 `0`。
- `time_window_violations`：时间窗违约数量。
- `late_minutes`：迟到分钟数。
- `total_cost`：距离、等待、迟到等合成成本。
- `route_count`：实际使用路线/车辆数量。

报告中建议先比较 `feasibility` 和 `cvr`，再比较 `total_cost`。

## 12. 最小验收流程

如果只想确认当前链路完整可用，按下面顺序执行：

```powershell
python -m py_compile dataset.py traffic.py evaluator.py decoder.py model.py heuristic.py inspect_data.py create_universal_split.py inspect_split.py train.py solve.py evaluate.py
python inspect_data.py --data_root C:\Users\86136\Desktop\code\RL\SVRP\solomon --sizes 100
python train.py --data_root C:\Users\86136\Desktop\code\RL\SVRP\solomon --size 100 --mode static --decoder strict_insert --insert_top_k 10 --imitation_epochs 1 --epochs 1 --steps_per_epoch 1 --batch_size 1 --val_limit 2 --checkpoint checkpoints\static_100_v2_smoke.pt
python evaluate.py --data_root C:\Users\86136\Desktop\code\RL\SVRP\solomon --sizes 100 --split test --decoder strict_insert --insert_top_k 10 --static_checkpoint checkpoints\static_100_v2_smoke_best.pt --mc_samples 2 --output_json results\static_100_v2_smoke_check.json --output_csv results\static_100_v2_smoke_check.csv
```

验收标准：

- 不出现 RL4CO/TorchRL/Lightning 相关报错。
- 能生成 `.pt` checkpoint。
- 能生成 JSON/CSV 结果。
- 静态结果中 `vehicles_excess=0.0`。
- 后续实验围绕 `strict_insert`、traffic-aware 训练和速度优化继续推进。

## 13. vrp_benchmark TW100 时间窗修复实验

`vrp_benchmark\real_twcvrp\twvrp_100_single_depot.npz` 可用于检查 Solomon 训练模型的跨数据源表现。当前支持两个额外开关：

- `--decoder deadline_aware_insert`：时间窗优先插入 decoder。
- `--post_opt time_window_repair`：在生成路线后尝试 relocate/swap 修复迟到客户。

先复现当前 baseline：

```powershell
python benchmark_evaluate.py --metric_profile native --mode static --decoder strict_insert --insert_top_k 0 --eval_set vrpbench_tw100_baseline --eval_files C:\Users\86136\Desktop\code\RL\SVRP\vrp_benchmark\real_twcvrp\twvrp_100_single_depot.npz --checkpoints base=checkpoints\static_100_v2_full_best.pt --output_json results\vrpbench_tw100_baseline_native.json --output_csv results\vrpbench_tw100_baseline_native.csv
```

测试 repair：

```powershell
python benchmark_evaluate.py --metric_profile native --mode static --decoder strict_insert --insert_top_k 0 --post_opt time_window_repair --eval_set vrpbench_tw100_repair --eval_files C:\Users\86136\Desktop\code\RL\SVRP\vrp_benchmark\real_twcvrp\twvrp_100_single_depot.npz --checkpoints base=checkpoints\static_100_v2_full_best.pt --output_json results\vrpbench_tw100_repair_native.json --output_csv results\vrpbench_tw100_repair_native.csv
```

测试 deadline-aware decoder：

```powershell
python benchmark_evaluate.py --metric_profile native --mode static --decoder deadline_aware_insert --insert_top_k 0 --post_opt time_window_repair --eval_set vrpbench_tw100_deadline_repair --eval_files C:\Users\86136\Desktop\code\RL\SVRP\vrp_benchmark\real_twcvrp\twvrp_100_single_depot.npz --checkpoints base=checkpoints\static_100_v2_full_best.pt --output_json results\vrpbench_tw100_deadline_repair_native.json --output_csv results\vrpbench_tw100_deadline_repair_native.csv
```

注意：本地诊断发现该 TW100 文件中存在若干客户从 depot 直接出发也超过 `due_time` 的情况，因此 native 口径下不一定存在 100% 硬可行解。报告中建议同时说明这一数据下界，避免把不可达时间窗误判为模型完全失效。

## 14. 交通扰动时间尺度 A/B 验证

为了验证 Solomon/Homberger 的时间尺度是否和 SVRP 的 24 小时交通扰动函数错位，可以只在评估阶段对比：

- `--traffic_time_scale raw`：直接使用路线当前时间，这是旧口径。
- `--traffic_time_scale depot_day`：把 depot 时间窗 `[ready_0, due_0]` 映射到 `[0, 1440]`，再计算早晚高峰和事故扰动。

验证该假设时建议先回到标准扰动强度，不使用强扰动版本：

```text
--traffic_profile additive --traffic_strength 1.0 --traffic_sigma 0.2 --traffic_buffer 0.5
```

100 规模 raw：

```powershell
python benchmark_evaluate.py --mode hybrid --metric_profile pdf_compatible --decoder strict_insert --insert_top_k 10 --post_opt none --split_root data_splits\universal_v1 --sizes 100 --split test --mc_samples 30 --traffic_seed 42 --traffic_profile additive --traffic_strength 1.0 --traffic_time_scale raw --checkpoints static100=checkpoints\static_100_v2_full_best.pt universal=checkpoints\tw_universal_all_full_best.pt --allow_unseen_size --output_json results\traffic_scale_raw_100.json --output_csv results\traffic_scale_raw_100.csv
```

100 规模 depot_day：

```powershell
python benchmark_evaluate.py --mode hybrid --metric_profile pdf_compatible --decoder strict_insert --insert_top_k 10 --post_opt none --split_root data_splits\universal_v1 --sizes 100 --split test --mc_samples 30 --traffic_seed 42 --traffic_profile additive --traffic_strength 1.0 --traffic_time_scale depot_day --checkpoints static100=checkpoints\static_100_v2_full_best.pt universal=checkpoints\tw_universal_all_full_best.pt --allow_unseen_size --output_json results\traffic_scale_depot_day_100.json --output_csv results\traffic_scale_depot_day_100.csv
```

对其它规模只替换 `--sizes` 和输出文件名即可，例如 `600`：

```powershell
python benchmark_evaluate.py --mode hybrid --metric_profile pdf_compatible --decoder strict_insert --insert_top_k 10 --post_opt none --split_root data_splits\universal_v1 --sizes 600 --split test --mc_samples 30 --traffic_seed 42 --traffic_profile additive --traffic_strength 1.0 --traffic_time_scale raw --checkpoints static100=checkpoints\static_100_v2_full_best.pt universal=checkpoints\tw_universal_all_full_best.pt --allow_unseen_size --output_json results\traffic_scale_raw_600.json --output_csv results\traffic_scale_raw_600.csv
python benchmark_evaluate.py --mode hybrid --metric_profile pdf_compatible --decoder strict_insert --insert_top_k 10 --post_opt none --split_root data_splits\universal_v1 --sizes 600 --split test --mc_samples 30 --traffic_seed 42 --traffic_profile additive --traffic_strength 1.0 --traffic_time_scale depot_day --checkpoints static100=checkpoints\static_100_v2_full_best.pt universal=checkpoints\tw_universal_all_full_best.pt --allow_unseen_size --output_json results\traffic_scale_depot_day_600.json --output_csv results\traffic_scale_depot_day_600.csv
```

CSV 中会额外输出：

- `traffic_time_scale`
- `avg_depot_due`
- `avg_raw_current_time`
- `avg_scaled_current_time`
- `avg_delay`
- `avg_delay_ratio`

如果 `raw` 与 `depot_day` 的 `avg_scaled_current_time`、`avg_delay_ratio`、`avg_cvr` 或 `feasibility_rate` 明显不同，说明时间尺度是交通扰动实验中的关键变量。
