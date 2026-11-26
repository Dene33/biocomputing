from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Task:
    name: str
    seconds: float
    command_fn: Callable[[float], tuple[float, float, float]]
    push_fn: Callable[[float], np.ndarray | None] | None = None


def _pushes(seed: int, every_s: float = 3.0, force_n: tuple[float, float] = (120.0, 200.0), dt: float = 0.02):
    rng = np.random.default_rng(seed)

    def push(t: float):
        if t > 1.0 and abs((t % every_s)) < dt / 2:
            angle = rng.uniform(0, 2 * np.pi)
            return rng.uniform(*force_n) * np.array([np.cos(angle), np.sin(angle)])
        return None

    return push


def _schedule(t: float):
    phases = [(2.0, (0.0, 0.0, 0.0)), (5.0, (0.4, 0.0, 0.0)), (4.0, (0.4, 0.0, 0.5)), (4.0, (0.2, 0.0, -0.5)),
              (5.0, (0.0, 0.0, 0.0))]
    for seconds, cmd in phases:
        if t < seconds:
            return cmd
        t -= seconds
    return (0.0, 0.0, 0.0)


def make_task(name: str, seconds: float = 20.0, seed: int = 0) -> Task:
    if name == "stand":
        return Task(name, seconds, lambda t: (0.0, 0.0, 0.0))
    if name == "stand_push":
        return Task(name, seconds, lambda t: (0.0, 0.0, 0.0), _pushes(seed))
    if name == "walk":
        return Task(name, seconds, lambda t: (0.0, 0.0, 0.0) if t < 1.0 else (0.4, 0.0, 0.0))
    if name == "walk_schedule":
        return Task(name, seconds, _schedule)
    raise ValueError(f"Unknown task {name!r}.")
