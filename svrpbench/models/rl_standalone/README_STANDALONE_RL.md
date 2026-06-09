# 独立 CVRP 强化学习链路说明

本文件夹提供一条独立的 PyTorch 强化学习路径规划链路，用于读取：

```text
vrp_benchmark/real_cvrp/*single_depot_single_vehicule_sumDemands.npz
```

它不依赖 RL4CO、TorchRL、Lightning、OR-Tools，也不使用：

```powershell
python -m vrp_bench solve
```

后续 CVRP 强化学习训练和推理优先使用本文件夹中的 `train.py` 与 `solve.py`。

## 文件说明

- `dataset.py`：读取 `vrp_benchmark` 中的 CVRP `.npz` 数据格式。
- `model.py`：轻量 pointer-style 路径选择策略网络。
- `train.py`：纯 PyTorch REINFORCE 训练入口。
- `solve.py`：加载 `.pt` checkpoint，贪心解码并导出路线结果。

## 环境要求

先手动进入 conda 环境：

```powershell
conda activate svrp
```

确认最小依赖可用：

```powershell
python -c "import torch, numpy, matplotlib; print(torch.__version__)"
```

如果缺少某个包，请在 `svrp` 环境中手动安装。该独立链路不需要安装 RL4CO、TorchRL 或 Lightning。

## Smoke 训练

进入独立 RL 文件夹：

```powershell
cd C:\Users\86136\Desktop\code\RL\SVRP\svrpbench\models\rl_standalone
```

执行 1 epoch 小规模训练，用于确认流程是否能跑通：

```powershell
python train.py --data ..\..\..\vrp_benchmark\real_cvrp\cvrp_10_single_depot_single_vehicule_sumDemands.npz --epochs 1 --batch_size 2 --checkpoint checkpoints\cvrp10.pt
```

成功后应看到类似输出：

```text
[DONE] checkpoint=checkpoints\cvrp10.pt
```

并生成：

```text
checkpoints/cvrp10.pt
checkpoints/cvrp10.json
```

其中 `.pt` 是模型权重，`.json` 是训练配置与日志摘要。

## 正常小规模训练

CVRP10 示例：

```powershell
python train.py --data ..\..\..\vrp_benchmark\real_cvrp\cvrp_10_single_depot_single_vehicule_sumDemands.npz --epochs 100 --batch_size 8 --checkpoint checkpoints\cvrp10.pt
```

更大规模数据需要训练对应客户数的 checkpoint。例如：

```powershell
python train.py --data ..\..\..\vrp_benchmark\real_cvrp\cvrp_20_single_depot_single_vehicule_sumDemands.npz --epochs 100 --batch_size 8 --checkpoint checkpoints\cvrp20.pt
python train.py --data ..\..\..\vrp_benchmark\real_cvrp\cvrp_50_single_depot_single_vehicule_sumDemands.npz --epochs 100 --batch_size 8 --checkpoint checkpoints\cvrp50.pt
```

如果显存不足，可以减小：

```powershell
--batch_size
```

## 推理生成路线

训练得到 checkpoint 后执行：

```powershell
python solve.py --data ..\..\..\vrp_benchmark\real_cvrp\cvrp_10_single_depot_single_vehicule_sumDemands.npz --checkpoint checkpoints\cvrp10.pt --output results\cvrp10_solutions.json --plot
```

成功后应看到类似输出：

```text
[DONE] output=results\cvrp10_solutions.json instances=10 feasible=... avg_distance=...
```

输出 JSON 位于：

```text
results/cvrp10_solutions.json
```

如果使用 `--plot`，路线图片会输出到：

```text
results/plots/
```

如果 Windows 下使用 `--plot` 出现 OpenMP 冲突：

```text
OMP: Error #15: Initializing libomp.dll, but found libiomp5md.dll already initialized.
```

建议先只生成 JSON，不在 `solve.py` 里画图：

```powershell
python solve.py --data ..\..\..\vrp_benchmark\real_cvrp\cvrp_10_single_depot_single_vehicule_sumDemands.npz --checkpoint checkpoints\cvrp10.pt --output results\cvrp10_solutions.json
```

然后用独立绘图脚本读取 JSON 画图。该脚本不导入 PyTorch，可以避免多数 OpenMP 冲突：

```powershell
python plot_results.py --data ..\..\..\vrp_benchmark\real_cvrp\cvrp_10_single_depot_single_vehicule_sumDemands.npz --solutions results\cvrp10_solutions.json --plot_dir results\plots
```

如果独立绘图脚本仍然报同样的 OpenMP 冲突，再使用兜底开关：

```powershell
python plot_results.py --data ..\..\..\vrp_benchmark\real_cvrp\cvrp_10_single_depot_single_vehicule_sumDemands.npz --solutions results\cvrp10_solutions.json --plot_dir results\plots --allow_duplicate_openmp
```

## 输出 JSON 格式

每个实例的结果类似：

```json
{
  "instance": 0,
  "source": "...",
  "routes": [[0, 3, 5, 1, 0]],
  "total_distance": 1234.5,
  "feasible": true,
  "capacity_used": [42.0],
  "served_customers": [1, 3, 5],
  "missing_customers": [],
  "duplicate_visits": 0,
  "vehicle_capacity": 120.0,
  "num_vehicles": 1
}
```

其中：

- `routes`：路线，节点 `0` 是 depot，客户节点从 `1` 开始。
- `total_distance`：路线总欧氏距离。
- `feasible`：是否没有重复访问、没有漏访，并且需求不超过可用容量。
- `capacity_used`：路线服务客户的总需求。
- `served_customers`：已服务客户编号。
- `missing_customers`：未服务客户编号。
- `duplicate_visits`：重复访问次数。

## 当前限制

- 第一版只支持 single-depot CVRP。
- 暂不支持 TWVRP。
- 暂不支持 multi-depot。
- 暂不做完整 multi-vehicle 路线拆分。
- 当前模型是轻量 baseline，目标是稳定打通“数据集 -> 强化学习训练 -> 路线规划结果”链路，不用于复现论文指标。

## 常见问题

### Checkpoint expects N customers

说明 checkpoint 的客户数和输入数据客户数不一致。例如 `cvrp10.pt` 只能用于 10 客户数据。

解决方法：为对应规模重新训练 checkpoint。

### standalone RL v1 supports only one depot

说明输入数据不是 single-depot CVRP 文件。

请使用类似：

```text
cvrp_10_single_depot_single_vehicule_sumDemands.npz
```

### matplotlib is required for --plot

说明使用了 `--plot`，但环境中缺少 matplotlib。

解决方法：

```powershell
pip install matplotlib
```

或者去掉 `--plot` 参数，只输出 JSON。
