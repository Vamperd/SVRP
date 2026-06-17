# Event-Driven Online Recourse TWCVRP 实验线

这是一条独立于 `rl_solomon_tw` 的在线追溯实验链路，用来验证“事件驱动再决策”是否能改善交通扰动和紧时间窗场景下的可行性。

核心思路：

- 不在 `t=0` 一次性生成完整客户排列。
- 每一步只为最早空闲车辆选择下一个客户。
- 动作空间为剩余客户数，而不是 `车辆数 x 客户数`。
- 默认 hard mask 已服务、容量超限、最快到达也必然迟到的客户。
- 如果所有客户都被时间窗 mask，会 fallback 到容量可行客户，并记录 `forced_late_actions`。
- 当前推荐使用 imitation warm-start：先模仿 `earliest_due`，再做 REINFORCE 微调。

## 1. 环境检查

```powershell
conda activate svrp
cd C:\Users\86136\Desktop\code\RL\SVRP\svrpbench\models\rl_recourse_tw
python -m py_compile common.py env.py policy.py rollout.py heuristic.py train.py evaluate.py
```

## 2. 启发式与 strict_insert 基线

先不训练，比较在线启发式与旧 `strict_insert`：

```powershell
python evaluate.py --split_root ..\rl_solomon_tw\data_splits\universal_v1 --sizes 100 --split test --mode traffic --traffic_sigma 0.2 --traffic_profile additive --traffic_strength 1.0 --traffic_time_scale depot_day --mc_samples 5 --methods earliest_due min_late strict_insert --strict_checkpoint ..\rl_solomon_tw\checkpoints\static_100_v2_full_best.pt --allow_unseen_size --output_json results\recourse_heuristic_100.json --output_csv results\recourse_heuristic_100.csv
```

如果 `earliest_due` 的 `avg_cvr` 和 `late_minutes` 明显低于 `strict_insert`，说明在线 recourse 机制有继续训练价值。

## 3. Imitation smoke 训练

```powershell
python train.py --split_root ..\rl_solomon_tw\data_splits\universal_v1 --sizes 100 --mode traffic --traffic_sigma 0.2 --traffic_profile additive --traffic_strength 1.0 --traffic_time_scale depot_day --risk_objective cvar --cvar_alpha 0.2 --mc_samples_train 1 --imitation_epochs 2 --expert_strategy earliest_due --imitation_weight 1.0 --bc_weight_after_imitation 0.05 --epochs 5 --steps_per_epoch 4 --batch_size 2 --val_every 1 --val_limit 4 --checkpoint checkpoints\recourse_100_imitation_smoke.pt
```

日志中应出现：

- `phase=imitation`
- `train_acc`
- `train_cvr`
- `val_cvr`

## 4. 100 规模正式训练

```powershell
python train.py --split_root ..\rl_solomon_tw\data_splits\universal_v1 --sizes 100 --mode traffic --traffic_sigma 0.2 --traffic_profile additive --traffic_strength 1.0 --traffic_time_scale depot_day --risk_objective cvar --cvar_alpha 0.2 --mc_samples_train 2 --imitation_epochs 10 --expert_strategy earliest_due --imitation_weight 1.0 --bc_weight_after_imitation 0.05 --epochs 40 --steps_per_epoch 12 --batch_size 2 --val_every 5 --val_limit 8 --lr 1e-4 --late_penalty 300 --time_window_penalty 10000 --forced_penalty 1000 --cost_weight 1.0 --checkpoint checkpoints\recourse_100_imitation.pt
```

## 5. 100 测试集对比

```powershell
python evaluate.py --split_root ..\rl_solomon_tw\data_splits\universal_v1 --sizes 100 --split test --mode traffic --traffic_sigma 0.2 --traffic_profile additive --traffic_strength 1.0 --traffic_time_scale depot_day --mc_samples 30 --methods earliest_due min_late strict_insert recourse --checkpoint checkpoints\recourse_100_imitation_best.pt --strict_checkpoint ..\rl_solomon_tw\checkpoints\static_100_v2_full_best.pt --allow_unseen_size --output_json results\recourse_100_imitation_test.json --output_csv results\recourse_100_imitation_test.csv
```

验收重点：

- `recourse avg_cvr` 应明显低于旧 recourse 的约 `7.18`。
- 理想情况下，`recourse avg_cvr <= strict_insert avg_cvr`。
- 如果 `recourse` 接近 `earliest_due` 的 CVR，但成本较高，说明 imitation 已有效，下一阶段再做成本压缩。

## 6. 100 模型跨规模评估

```powershell
python evaluate.py --split_root ..\rl_solomon_tw\data_splits\universal_v1 --sizes 200 --split test --mode traffic --traffic_sigma 0.2 --traffic_profile additive --traffic_strength 1.0 --traffic_time_scale depot_day --mc_samples 30 --methods earliest_due min_late strict_insert recourse --checkpoint checkpoints\recourse_100_imitation_best.pt --strict_checkpoint ..\rl_solomon_tw\checkpoints\static_100_v2_full_best.pt --allow_unseen_size --output_json results\recourse_100_imitation_on_200.json --output_csv results\recourse_100_imitation_on_200.csv
```

```powershell
python evaluate.py --split_root ..\rl_solomon_tw\data_splits\universal_v1 --sizes 400 --split test --mode traffic --traffic_sigma 0.2 --traffic_profile additive --traffic_strength 1.0 --traffic_time_scale depot_day --mc_samples 30 --methods earliest_due min_late strict_insert recourse --checkpoint checkpoints\recourse_100_imitation_best.pt --strict_checkpoint ..\rl_solomon_tw\checkpoints\static_100_v2_full_best.pt --allow_unseen_size --output_json results\recourse_100_imitation_on_400.json --output_csv results\recourse_100_imitation_on_400.csv
```

## 7. 可选：多规模训练

只有当 100/200/400 的 imitation 结果明显好于旧 recourse 后，再运行：

```powershell
python train.py --split_root ..\rl_solomon_tw\data_splits\universal_v1 --sizes 100 200 400 --mode traffic --traffic_sigma 0.2 --traffic_profile additive --traffic_strength 1.0 --traffic_time_scale depot_day --risk_objective cvar --cvar_alpha 0.2 --mc_samples_train 2 --imitation_epochs 10 --expert_strategy earliest_due --imitation_weight 1.0 --bc_weight_after_imitation 0.05 --epochs 50 --steps_per_epoch 12 --batch_size 2 --val_every 5 --val_limit 8 --checkpoint checkpoints\recourse_100_200_400_imitation.pt
```

## 指标解释

- `avg_cvr`：约束违约率，越低越好。
- `feasibility_rate`：完全可行率，越高越好。
- `late_minutes`：平均迟到分钟数。
- `forced_late_actions`：hard mask 后仍不得不选择迟到客户的次数，越低越好。
- `route_count`：使用路线数，避免为了可行性过度膨胀。

当前阶段第一目标是让神经 recourse 学会 `earliest_due` 的低违约行为；成本优化放在下一阶段。
