"""RL4CO solver adapters for the SVRPBench registry."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

from vrp_bench.core import Instance, Solution, Solver, register_solver
from vrp_bench.solvers._legacy import _LEGACY_DIR  # Adds legacy modules to sys.path.

from vrp_base import VRPSolverBase  # noqa: E402

_SVRPBENCH_ROOT = Path(__file__).resolve().parents[2]
if str(_SVRPBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_SVRPBENCH_ROOT))

from models.rl import utils as rl_utils  # noqa: E402


class _MetricCalculator(VRPSolverBase):
    """Small concrete wrapper so RL routes can reuse benchmark metrics."""

    def solve_instance(self, instance_idx: int, num_realizations: int = 3) -> dict:
        raise NotImplementedError("RL routes are supplied directly to calculate_solution_cost.")


class RL4COSolver(Solver):
    """Base class for RL4CO checkpoints used as benchmark solvers."""

    algo: str = ""

    def __init__(
        self,
        *,
        checkpoint_root: Optional[str | Path] = None,
        device: Optional[str] = None,
        decode_type: str = "greedy",
        map_size: Optional[float] = None,
        max_time: float = 1440.0,
    ):
        default_root = _SVRPBENCH_ROOT / "models" / "rl" / "checkpoints"
        self.checkpoint_root = Path(
            checkpoint_root
            or os.environ.get("SVRP_RL_CHECKPOINT_ROOT")
            or default_root
        )
        self.device = device
        self.decode_type = decode_type
        self.map_size = map_size
        self.max_time = max_time

    def solve(self, instance: Instance, *, num_realizations: int = 1) -> Solution:
        started = time.time()
        rl_utils.ensure_single_depot(instance)
        variant = rl_utils.infer_variant(instance)
        num_loc = rl_utils.num_customers(instance)

        torch, _ = rl_utils.require_tensor_libs()
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, checkpoint_path = self._load_model(variant=variant, num_loc=num_loc, device=device)
        policy, env = model.policy, model.env
        policy = policy.to(device)
        policy.eval()

        td = rl_utils.instance_to_tensordict(
            instance,
            variant=variant,
            map_size=self.map_size,
            max_time=self.max_time,
        ).to(device)

        try:
            td_init = env.reset(td=td.clone(), batch_size=[1]).to(device)
        except TypeError:
            td_init = env.reset(td=td.clone()).to(device)

        with torch.no_grad():
            try:
                out = policy(td_init.clone(), env, phase="test", decode_type=self.decode_type)
            except TypeError:
                out = policy(td_init.clone(), env, phase="test")

        actions = rl_utils.best_action_sequence(out)
        routes = rl_utils.actions_to_routes(actions, num_nodes=instance.num_nodes)
        metrics = self._score_routes(instance, routes, num_realizations)
        metrics["runtime"] = time.time() - started
        metrics["checkpoint"] = str(checkpoint_path)
        metrics["variant"] = variant
        metrics["algo"] = self.algo

        solution_metrics = dict(metrics)
        solution_metrics.pop("routes", None)
        return Solution.from_metrics(solution_metrics, routes=routes)

    def _load_model(self, *, variant: str, num_loc: int, device: str):
        try:
            from rl4co.models import POMO, REINFORCE
        except ModuleNotFoundError as exc:
            raise RuntimeError(rl_utils.RL_DEPENDENCY_MESSAGE) from exc

        model_cls = REINFORCE if self.algo == "attention" else POMO
        checkpoint_path = rl_utils.resolve_checkpoint(
            self.checkpoint_root,
            variant=variant,
            algo=self.algo,
            num_loc=num_loc,
        )
        import torch

        original_torch_load = torch.load

        def trusted_checkpoint_load(*args, **kwargs):
            if kwargs.get("weights_only") is None:
                kwargs["weights_only"] = False
            return original_torch_load(*args, **kwargs)

        try:
            torch.load = trusted_checkpoint_load
            model = model_cls.load_from_checkpoint(str(checkpoint_path), map_location=device)
        finally:
            torch.load = original_torch_load
        model = model.to(device)
        model.eval()
        return model, checkpoint_path

    def _score_routes(
        self,
        instance: Instance,
        routes: list[list[int]],
        num_realizations: int,
    ) -> dict:
        calculator = _MetricCalculator(instance.to_legacy_dict())
        metrics = calculator.calculate_solution_cost(
            routes,
            instance_idx=0,
            num_realizations=num_realizations,
        )

        used_routes = [route for route in routes if len(route) > 2]
        overflow = max(0, len(used_routes) - int(instance.num_vehicles))
        if overflow:
            metrics["vehicle_count_violations"] = overflow
            metrics["feasibility"] = 0.0
            customer_count = max(1, rl_utils.num_customers(instance))
            metrics["cvr"] = max(
                float(metrics.get("cvr", 0.0)),
                (overflow / customer_count) * 100.0,
            )

        return metrics


@register_solver("attention")
class AttentionModelSolver(RL4COSolver):
    """Attention Model checkpoint trained with REINFORCE."""

    algo = "attention"


@register_solver("pomo")
class POMOSolver(RL4COSolver):
    """POMO checkpoint trained with a shared baseline."""

    algo = "pomo"
