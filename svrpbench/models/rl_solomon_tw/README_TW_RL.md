# 独立 Solomon/TWCVRP 强化学习链路说明

本文件夹实现一条独立 PyTorch 强化学习链路，用于形成 `benchmark_report.pdf` 的 RL 对照组。它不依赖 RL4CO、TorchRL、Lightning，也不调用 `python -m vrp_bench solve`。

## 1. 环境检查

所有命令都由你在 conda `svrp` 环境中手动执行：

```powershell
conda activate svrp
cd C:\Users\86136\Desktop\code\RL\SVRP\svrpbench\models\rl_solomon_tw
python -c "import torch, numpy; print(torch.__version__)"
```

如果缺少依赖，手动安装最小依赖：

```powershell
pip install torch numpy
```

## 2. 数据来源

优先使用本地 Solomon/Homberger 文本数据：

```text
C:\Users\86136\Desktop\code\RL\SVRP\solomon
```

当前已识别规模：

```text
100, 200, 400, 600, 800, 1000
```

也可以读取 `vrp_benchmark\real_twcvrp` 中的 `.npz`，但报告对照建议优先使用 Solomon 文件夹。

## 3. 检查数据与划分

```powershell
python inspect_data.py --data_root C:\Users\86136\Desktop\code\RL\SVRP\solomon --sizes 100 200 400 600 800 1000
```

默认划分：

- `train_ratio=0.70`
- `val_ratio=0.15`
- 剩余为 test
- `seed=1234`
- 按 `C1/C2/R1/R2/RC1/RC2` 分层划分

## 4. 可选：手动生成交通扰动说明或缓存

Solomon 原始文件不包含交通扰动矩阵，因此交通扰动由脚本合成生成。

默认只生成 manifest，不保存完整矩阵，适合先检查配置：

```powershell
python prepare_traffic.py --data_root C:\Users\86136\Desktop\code\RL\SVRP\solomon --sizes 100 --mc_samples 30 --traffic_sigma 0.2 --output_dir data_cache\traffic
```

如果确实需要保存完整矩阵，使用：

```powershell
python prepare_traffic.py --data_root C:\Users\86136\Desktop\code\RL\SVRP\solomon --sizes 100 --mc_samples 30 --traffic_sigma 0.2 --output_dir data_cache\traffic --store full
```

注意：`--store full` 在 1000 规模会非常占磁盘，因为每个样本矩阵约为 `1001 x 1001`。

训练和评估脚本默认会按固定 seed 在线生成扰动矩阵，不强制依赖缓存。

## 5. 启发式 sanity check

先用最近邻启发式确认静态 TWCVRP 指标能算通：

```powershell
python heuristic.py --input C:\Users\86136\Desktop\code\RL\SVRP\solomon\100\c101.txt --mode static --output results\heuristic_c101.json
```

## 6. 静态 RL smoke 训练

```powershell
python train.py --data_root C:\Users\86136\Desktop\code\RL\SVRP\solomon --size 100 --mode static --epochs 1 --batch_size 2 --checkpoint checkpoints\static_100.pt --smoke
```

正常小规模训练：

```powershell
python train.py --data_root C:\Users\86136\Desktop\code\RL\SVRP\solomon --size 100 --mode static --epochs 100 --batch_size 8 --checkpoint checkpoints\static_100.pt
```

## 7. 交通扰动 RL smoke 训练

```powershell
python train.py --data_root C:\Users\86136\Desktop\code\RL\SVRP\solomon --size 100 --mode traffic --epochs 1 --batch_size 2 --traffic_sigma 0.2 --checkpoint checkpoints\traffic_100.pt --smoke
```

正常小规模训练：

```powershell
python train.py --data_root C:\Users\86136\Desktop\code\RL\SVRP\solomon --size 100 --mode traffic --epochs 100 --batch_size 8 --traffic_sigma 0.2 --checkpoint checkpoints\traffic_100.pt
```

## 8. 单实例推理与 SVG 绘图

```powershell
python solve.py --input C:\Users\86136\Desktop\code\RL\SVRP\solomon\100\c101.txt --checkpoint checkpoints\static_100.pt --mode static --output results\c101_static_solution.json --plot_svg results\plots\c101_static.svg
```

SVG 绘图不依赖 Matplotlib，通常不会触发 OpenMP DLL 冲突。

## 9. 批量对照评估

只评估 100 规模 test 集：

```powershell
python evaluate.py --data_root C:\Users\86136\Desktop\code\RL\SVRP\solomon --sizes 100 --static_checkpoint checkpoints\static_100.pt --traffic_checkpoint checkpoints\traffic_100.pt --split test --mc_samples 30 --output_json results\comparison_100.json --output_csv results\comparison_100.csv
```

多个规模建议每个规模训练独立 checkpoint，然后用模板路径：

```powershell
python evaluate.py --data_root C:\Users\86136\Desktop\code\RL\SVRP\solomon --sizes 100 200 400 600 800 1000 --static_checkpoint checkpoints\static_{size}.pt --traffic_checkpoint checkpoints\traffic_{size}.pt --split test --mc_samples 30 --output_json results\comparison_all.json --output_csv results\comparison_all.csv
```

## 10. 验收标准

- `python -m py_compile *.py` 通过。
- `inspect_data.py` 能识别所有 Solomon 规模。
- smoke 训练能生成 `.pt` checkpoint 和 `.json` 元数据。
- `solve.py` 能生成路线 JSON 和 SVG。
- `evaluate.py` 能生成 `static_rl`、`hybrid_rl`、`traffic_rl` 三组结果。
- 输出中包含 `cvr`、`feasibility`、`runtime`、`robustness_std`。

## 10.1 可行性优先 v2 策略

如果旧训练出现 `vehicles_excess` 很高、`feasibility` 长期不提升，使用 v2 策略：

- 默认 decoder 为 `strict_insert`，最多只使用实例允许的车辆数。
- reward 优先奖励可行解、惩罚不可行解，再比较成本。
- best checkpoint 按 `val_feasibility` 最大、`val_cvr` 最小、`val_total_cost` 最小选择。
- 支持 `--imitation_epochs`，先模仿时间窗最近邻启发式，再进入 REINFORCE。

推荐 100 规模训练命令：

```powershell
python train.py --data_root C:\Users\86136\Desktop\code\RL\SVRP\solomon --size 100 --mode static --decoder strict_insert --imitation_epochs 20 --epochs 300 --steps_per_epoch 20 --batch_size 8 --feasible_bonus 50000 --infeasible_penalty 50000 --vehicle_penalty 10000 --route_count_penalty 200 --time_window_penalty 5000 --capacity_penalty 5000 --late_penalty 50 --checkpoint checkpoints\static_100_v2.pt
```

测试：

```powershell
python evaluate.py --data_root C:\Users\86136\Desktop\code\RL\SVRP\solomon --sizes 100 --split test --decoder strict_insert --static_checkpoint checkpoints\static_100_v2_best.pt --mc_samples 30 --output_json results\static_100_v2_test.json --output_csv results\static_100_v2_test.csv
```

和旧 decoder 做对照：

```powershell
python evaluate.py --data_root C:\Users\86136\Desktop\code\RL\SVRP\solomon --sizes 100 --split test --decoder greedy_split --static_checkpoint checkpoints\static_100_feasible_best.pt --mc_samples 30 --output_json results\static_100_greedy_split_test.json --output_csv results\static_100_greedy_split_test.csv
```

## 11. 通用多规模模型 universal_v1

如果希望训练一个 checkpoint 同时求解 `100/200/400/600/800/1000`，使用 `data_splits/universal_v1` 文件级划分。

目录结构：

```text
data_splits/universal_v1/
  split_manifest.json
  train/100
  train/200
  train/400
  train/600
  train/800
  train/1000
  val/...
  test/...
```

生成通用划分：

```powershell
python create_universal_split.py --source_root C:\Users\86136\Desktop\code\RL\SVRP\solomon --output_root data_splits\universal_v1
```

如果该目录已经存在且需要重新复制，请显式使用：

```powershell
python create_universal_split.py --source_root C:\Users\86136\Desktop\code\RL\SVRP\solomon --output_root data_splits\universal_v1 --overwrite
```

检查划分数量：

```powershell
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

静态通用 smoke 训练：

```powershell
python train.py --split_root data_splits\universal_v1 --sizes 100 200 400 600 800 1000 --mode static --decoder strict_insert --epochs 1 --batch_size 2 --checkpoint checkpoints\tw_universal_static.pt --smoke
```

交通扰动通用 smoke 训练：

```powershell
python train.py --split_root data_splits\universal_v1 --sizes 100 200 400 600 800 1000 --mode traffic --decoder strict_insert --epochs 1 --batch_size 2 --traffic_sigma 0.2 --checkpoint checkpoints\tw_universal_traffic.pt --smoke
```

正式训练示例：

```powershell
python train.py --split_root data_splits\universal_v1 --sizes 100 200 400 600 800 1000 --mode static --decoder strict_insert --imitation_epochs 20 --epochs 100 --batch_size 8 --checkpoint checkpoints\tw_universal_static.pt
python train.py --split_root data_splits\universal_v1 --sizes 100 200 400 600 800 1000 --mode traffic --decoder strict_insert --imitation_epochs 20 --epochs 100 --batch_size 8 --traffic_sigma 0.2 --checkpoint checkpoints\tw_universal_traffic.pt
```

如果 800 或 1000 规模训练显存不足，先降低：

```powershell
--batch_size 1
```

测试集评估：

```powershell
python evaluate.py --split_root data_splits\universal_v1 --sizes 100 200 400 600 800 1000 --split test --decoder strict_insert --static_checkpoint checkpoints\tw_universal_static.pt --traffic_checkpoint checkpoints\tw_universal_traffic.pt --mc_samples 30 --output_json results\universal_test.json --output_csv results\universal_test.csv
```

单实例推理：

```powershell
python solve.py --input data_splits\universal_v1\test\100\c101.txt --checkpoint checkpoints\tw_universal_static.pt --mode static --decoder strict_insert --output results\single_solution.json --plot_svg results\plots\single_solution.svg
```

通用 checkpoint 会保存 `supported_sizes`。如果要测试未列入训练范围的规模，需要显式添加：

```powershell
--allow_unseen_size
```
