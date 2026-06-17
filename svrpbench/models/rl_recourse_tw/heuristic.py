"""Event-driven online recourse heuristics."""
from __future__ import annotations

from env import EventDrivenTWEnv


def run_heuristic(env: EventDrivenTWEnv, *, strategy: str = "earliest_due") -> dict:
    while not env.done:
        state = env.decision_state()
        if state is None:
            break
        if strategy == "earliest_due":
            customer = _earliest_due(env, state)
        elif strategy == "min_late":
            customer = _min_late(env, state)
        else:
            raise ValueError("strategy must be 'earliest_due' or 'min_late'.")
        env.step(customer, state)
    metrics = env.metrics()
    metrics["method"] = strategy
    return metrics


def _earliest_due(env: EventDrivenTWEnv, state) -> int:
    candidates = _candidate_nodes(state)
    return min(candidates, key=lambda node: (float(env.instance.due_times[node]), node))


def _min_late(env: EventDrivenTWEnv, state) -> int:
    current = int(state.current_node)
    current_time = float(state.current_time)

    def key(node: int) -> tuple[float, float, float, int]:
        arrival = current_time + float(env.travel[current, node])
        late = max(0.0, arrival - float(env.instance.due_times[node]))
        wait = max(0.0, float(env.instance.ready_times[node]) - arrival)
        return late, wait, float(env.travel[current, node]), node

    return min(_candidate_nodes(state), key=key)


def _candidate_nodes(state) -> list[int]:
    nodes = [idx for idx, allowed in enumerate(state.legal_mask) if idx > 0 and allowed]
    if nodes:
        return nodes
    return [idx for idx, allowed in enumerate(state.relaxed_mask) if idx > 0 and allowed]

