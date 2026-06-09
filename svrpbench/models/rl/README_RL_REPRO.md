# SVRPBench RL Reproduction Manual

本文档说明如何在 conda 的 `svrp` 环境中手动安装依赖、训练 RL4CO 模型，并把训练得到的 checkpoint 接入 `vrp_bench` benchmark。Codex 不会自动安装环境，也不会自动运行训练或评估命令。

## 1. 进入 conda 环境

在 PowerShell 中执行：

```powershell
conda env list
conda activate svrp
python --version
where python
```

如果 `svrp` 环境不存在，请手动创建：

```powershell
conda create -n svrp python=3.10
conda activate svrp
```

## 2. 安装依赖

先进入项目包目录：

```powershell
cd C:\Users\86136\Desktop\code\RL\SVRP\svrpbench
```

如果当前环境还没有 PyTorch，请先按本机 CUDA/CPU 情况从 PyTorch 官网选择安装命令：

```powershell
# 访问 https://pytorch.org/get-started/locally/ 后复制适合本机的命令
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

然后安装 RL 复现依赖：

```powershell
pip install -r models\rl\requirements-rl.txt
```

验证依赖：

```powershell
python -c "import rl4co, torch, lightning, tensordict; print('RL env ok', torch.__version__, torch.cuda.is_available())"
```

如果出现下面这种错误：

```text
ImportError: cannot import name 'BoundedTensorSpec' from 'torchrl.data'
```

说明 `rl4co==0.4.0` 装到了过新的 `torchrl`。RL4CO 0.4.0 的依赖范围较宽，而新版 TorchRL 已经把旧的 `BoundedTensorSpec` 名称迁移为新版 spec 名称。请在 `svrp` 环境中手动固定兼容版本：

```powershell
conda activate svrp
pip install --force-reinstall --no-deps torchrl==0.6.0 tensordict==0.6.0
python -c "from torchrl.data import BoundedTensorSpec; import torchrl, tensordict; print(torchrl.__version__, tensordict.__version__)"
```

如果你使用的是 PyTorch `cu124`，通常可以和 `torchrl==0.6.0` / `tensordict==0.6.0` 搭配；若仍有二进制兼容问题，请先打印版本：

```powershell
python -c "import torch, torchrl, tensordict; print(torch.__version__, torchrl.__version__, tensordict.__version__)"
```

## 3. Smoke 训练

这个步骤只验证流程是否闭环，不用于复现论文指标。

```powershell
cd C:\Users\86136\Desktop\code\RL\SVRP\svrpbench\models\rl
python train.py --variant cvrp --algo attention --num_loc 10 --batch_size 8 --max_epochs 1 --train_data_size 32 --val_data_size 8 --accelerator auto --smoke
```

成功后应生成类似目录：

```text
C:\Users\86136\Desktop\code\RL\SVRP\svrpbench\models\rl\checkpoints\cvrp\attention_10\
```

里面应包含 `last.ckpt` 或 `epoch_*.ckpt`。

## 4. Benchmark 推理

回到 `svrpbench` 目录：

```powershell
cd C:\Users\86136\Desktop\code\RL\SVRP\svrpbench
```

先确认 solver 注册：

```powershell
python -m vrp_bench list
```

应能看到：

```text
attention
pomo
```

使用 smoke checkpoint 对 1 个 CVRP 实例推理：

```powershell
python -m vrp_bench solve --solver attention --data ..\vrp_benchmark\real_cvrp\cvrp_10_single_depot_single_vehicule_sumDemands.npz --limit 1 --realizations 1
```

如果输出的 aggregate JSON 包含以下字段，即说明训练 checkpoint、RL solver、路线转换和 benchmark 指标计算已连通：

```text
total_cost
runtime
feasibility
cvr
robustness
```

## 5. 完整训练建议

论文一致的 v1 范围是 single-depot CVRP 与 single-depot TWVRP。建议先从小规模开始，确认每个 size 能保存 checkpoint，再扩大训练。

CVRP Attention/REINFORCE 示例：

```powershell
cd C:\Users\86136\Desktop\code\RL\SVRP\svrpbench\models\rl
python train.py --variant cvrp --algo attention --num_loc 10 --batch_size 512 --max_epochs 10 --accelerator auto
python train.py --variant cvrp --algo attention --num_loc 20 --batch_size 256 --max_epochs 10 --accelerator auto
python train.py --variant cvrp --algo attention --num_loc 50 --batch_size 128 --max_epochs 10 --accelerator auto
```

TWVRP POMO 示例：

```powershell
cd C:\Users\86136\Desktop\code\RL\SVRP\svrpbench\models\rl
python train.py --variant twvrp --algo pomo --num_loc 10 --batch_size 64 --max_epochs 10 --accelerator auto
python train.py --variant twvrp --algo pomo --num_loc 20 --batch_size 64 --max_epochs 10 --accelerator auto
python train.py --variant twvrp --algo pomo --num_loc 50 --batch_size 64 --max_epochs 10 --accelerator auto
```

可按同样格式扩展到 `100, 200, 500`。如果显存不足，优先减小 `--batch_size`。

注意：`rl4co==0.4.0` 的 PyPI 包可能不包含 `rl4co.envs.routing.mtvrp`。如果 TWVRP 训练时报：

```text
No module named 'rl4co.envs.routing.mtvrp'
```

说明当前 RL4CO 版本只能先跑 CVRP 训练。MTVRP/VRPTW 在 RL4CO v0.5.0 及更新文档中可见；若要训练 TWVRP，需要后续单独升级并重新校准 `torchrl/tensordict` 兼容版本。

## 6. 指定 checkpoint 目录

默认目录是：

```text
svrpbench\models\rl\checkpoints
```

如果要使用其它位置，可在运行 benchmark 前设置环境变量：

```powershell
$env:SVRP_RL_CHECKPOINT_ROOT="C:\path\to\checkpoints"
python -m vrp_bench solve --solver attention --data ..\vrp_benchmark\real_cvrp\cvrp_10_single_depot_single_vehicule_sumDemands.npz --limit 1 --realizations 1
```

目录结构必须保持：

```text
{checkpoint_root}\cvrp\attention_10\last.ckpt
{checkpoint_root}\twvrp\pomo_10\last.ckpt
```

## 7. 常见问题

- `ModuleNotFoundError: rl4co`  
  说明当前 shell 没有进入 `svrp` 环境，或还没有执行 `pip install -r models\rl\requirements-rl.txt`。

- `ImportError: cannot import name 'BoundedTensorSpec' from 'torchrl.data'`  
  说明 `torchrl` 版本过新。执行：
  `pip install --force-reinstall --no-deps torchrl==0.6.0 tensordict==0.6.0`。

- `ValueError: Either a dictionary or a sequence of kwargs must be provided, not both.`  
  这是 `rl4co==0.4.0` 与新版 TensorDict 构造参数之间的兼容问题。本仓库的 `train.py` 已在运行时 patch RL4CO 的 dataset collate 逻辑；请确认正在使用最新的 `train.py` 后重新运行训练命令。

- `No module named 'rl4co.envs.routing.mtvrp'`  
  当前安装的 RL4CO 不包含 MTVRP/VRPTW 环境。CVRP smoke 和 CVRP 训练仍可继续；TWVRP 训练需要后续升级 RL4CO 并重新处理依赖兼容。

- `No RL checkpoint found`  
  说明 solver 找不到 `last.ckpt` 或 `epoch_*.ckpt`。先完成对应 `variant/algo/num_loc` 的训练，或设置 `SVRP_RL_CHECKPOINT_ROOT`。

- `_pickle.UnpicklingError: Weights only load failed`  
  PyTorch 2.6 起 `torch.load` 默认使用 `weights_only=True`。本仓库的 RL solver 会对本机训练得到的可信 checkpoint 显式使用 `weights_only=False` 加载；请确认正在使用最新的 `vrp_bench/solvers/rl.py` 后重试。

- `RL reproduction v1 supports exactly one depot`  
  当前 RL 复现只支持 single-depot 文件。请使用 `real_cvrp\*_single_depot_single_vehicule_sumDemands.npz` 或 `real_twcvrp\*_single_depot.npz`。

- CUDA 不可用  
  运行 `python -c "import torch; print(torch.cuda.is_available())"` 检查。若为 `False`，重新安装匹配本机 CUDA 的 PyTorch wheel，或使用较小 batch 在 CPU 上做 smoke test。
