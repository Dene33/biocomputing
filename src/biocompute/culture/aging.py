"""Nonstationarity: a real culture is not the same from one day to the next."""

from __future__ import annotations

import torch


def age(culture, days: float, drift_per_day: float = 0.5, death_per_day: float = 0.02, seed: int = 0) -> None:
    """Advance a culture by ``days``: excitability drifts as a random walk (current units per sqrt(day)) and a
    fraction of the neurons dies each day."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    step = drift_per_day * (days**0.5) * torch.randn(culture.n, generator=g)
    culture.excitability += step.to(culture.device)
    culture.kill(1.0 - (1.0 - death_per_day) ** days)
