# CVRP 与 TWCVRP 强化学习训练、验证、测试策略

本文说明两个独立强化学习链路如何划分训练集、验证集、测试集，以及如何做检测与对照。

## 共同原则

- 不使用测试集参与训练、调参或选择 checkpoint。
- 固定 `seed=1234` 做数据划分，固定 `traffic_seed=42` 做交通扰动评估。
- 训练集用于更新模型；验证集用于观察训练过程和选择 best checkpoint；测试集只用于最终报告。
- 每个客户规模单独训练更公平，例如 CVRP10 用 CVRP10 checkpoint，TWCVRP100 用 TWCVRP100 checkpoint。
- 跨规模测试可以作为泛化观察，但不作为主结果。

## CVRP 链路

位置：

```text
C:\Users\86136\Desktop\code\RL\SVRP\svrpbench\models\rl_standalone
```

数据：

```text
C:\Users\86136\Desktop\code\RL\SVRP\vrp_benchmark\real_cvrp
```

默认划分方式：

- 对 `.npz` 文件中的实例按 index 做确定性 shuffle。
- 默认比例为 `70% train / 15% val / 15% test`。
- 以只有 10 个实例的 CVRP10 文件为例，通常得到约 `7 train / 1 val / 2 test`。

Smoke 训练仍可使用全量数据，确认链路是否跑通：

```powershell
conda activate svrp
cd C:\Users\86136\Desktop\code\RL\SVRP\svrpbench\models\rl_standalone
python train.py --data ..\..\..\vrp_benchmark\real_cvrp\cvrp_10_single_depot_single_vehicule_sumDemands.npz --epochs 1 --batch_size 2 --checkpoint checkpoints\cvrp10.pt
```

正式训练建议打开 split：

```powershell
python train.py --data ..\..\..\vrp_benchmark\real_cvrp\cvrp_10_single_depot_single_vehicule_sumDemands.npz --epochs 100 --batch_size 8 --checkpoint checkpoints\cvrp10.pt --use_split --split_seed 1234 --train_ratio 0.70 --val_ratio 0.15
```

测试集推理：

```powershell
python solve.py --data ..\..\..\vrp_benchmark\real_cvrp\cvrp_10_single_depot_single_vehicule_sumDemands.npz --checkpoint checkpoints\cvrp10.pt --split test --split_seed 1234 --output results\cvrp10_test_solutions.json
```

检测重点：

- JSON 中每条 route 从 `0` 出发并回到 `0`。
- `missing_customers` 为空。
- `duplicate_visits` 为 `0`。
- `feasible=true` 的比例作为可行率。
- `total_distance` 作为主要成本。

## TWCVRP 链路

位置：

```text
C:\Users\86136\Desktop\code\RL\SVRP\svrpbench\models\rl_solomon_tw
```

静态数据优先使用：

```text
C:\Users\86136\Desktop\code\RL\SVRP\solomon
```

兼容输入：

```text
C:\Users\86136\Desktop\code\RL\SVRP\vrp_benchmark\real_twcvrp
```

默认划分方式：

- Solomon/Homberger 按文件划分，而不是按客户节点划分。
- 默认按问题族分层：`C1/C2/R1/R2/RC1/RC2` 各自做 deterministic shuffle。
- 默认比例为 `70% train / 15% val / 15% test`。
- `solomon\100` 有 56 个文件，约为 `39 train / 8 val / 9 test`。
- `solomon\200/400/600/800/1000` 各有 60 个文件，约为 `42 train / 9 val / 9 test`。

实验分三组：

- `static_rl`：静态矩阵训练，静态矩阵推理，静态矩阵测试。
- `hybrid_rl`：静态矩阵训练和推理，交通扰动 Monte Carlo 测试。
- `traffic_rl`：交通扰动训练，交通感知矩阵推理，交通扰动 Monte Carlo 测试。

检测指标：

- `total_distance`
- `total_travel_time`
- `waiting_time`
- `late_minutes`
- `time_window_violations`
- `capacity_violations`
- `route_count`
- `cvr`
- `feasibility`
- `runtime`
- `robustness_std`

## 推荐实验顺序

1. CVRP10 smoke，确认旧独立链路仍可用。
2. TWCVRP100 数据检查，确认 Solomon 文件可解析。
3. TWCVRP100 heuristic sanity check，确认时间窗指标计算正常。
4. TWCVRP100 static smoke 训练。
5. TWCVRP100 traffic smoke 训练。
6. TWCVRP100 test split 批量评估。
7. 扩展到 200/400/600/800/1000，每个规模单独训练 checkpoint。

## TWCVRP 多规模通用模型

如果目标是训练一个模型同时求解 `100/200/400/600/800/1000`，使用：

```text
C:\Users\86136\Desktop\code\RL\SVRP\svrpbench\models\rl_solomon_tw\data_splits\universal_v1
```

该划分按文件复制 Solomon 原始数据，不删除、不移动原始 `solomon` 文件。

划分规则：

- `100/200/400` 是主训练规模，使用常规 `70% train / 15% val / 15% test`。
- `600/800/1000` 是大规模适应规模，每个规模只放 `12` 个文件进 train。
- 所有规模都保留 val，用于选择 best checkpoint。
- 所有规模都保留完全未见过的 test，用于最终报告。

预期数量：

```text
100:  39 train / 8 val / 9 test
200:  42 train / 9 val / 9 test
400:  42 train / 9 val / 9 test
600:  12 train / 9 val / 39 test
800:  12 train / 9 val / 39 test
1000: 12 train / 9 val / 39 test
```

通用模型训练方式：

- batch 内保持同一规模，避免 padding 和无效计算。
- 每个 epoch 默认均衡采样所有规模。
- checkpoint 保存 `supported_sizes`，推理时允许同一个模型处理多个规模。
- `static_rl`、`hybrid_rl`、`traffic_rl` 仍然按 test split 输出，不能使用 train 或 val 结果代替最终测试。
- 当前 v2 策略默认使用 `strict_insert` decoder，优先保证不超过车辆数。
- best checkpoint 按 `val_feasibility`、`val_cvr`、`val_total_cost` 的顺序选择。
- 若旧模型出现 `vehicles_excess` 高、可行率低，应优先使用 v2 可行性训练命令，而不是只比较 `total_cost`。

推荐命令见：

```text
C:\Users\86136\Desktop\code\RL\SVRP\svrpbench\models\rl_solomon_tw\README_TW_RL.md
```
