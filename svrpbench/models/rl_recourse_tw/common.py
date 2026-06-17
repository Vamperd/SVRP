"""Shared imports and helpers for the online recourse TWCVRP experiment line."""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOLOMON_TW_DIR = Path(__file__).resolve().parents[1] / "rl_solomon_tw"
if str(SOLOMON_TW_DIR) not in sys.path:
    sys.path.append(str(SOLOMON_TW_DIR))


def ensure_dir(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output
