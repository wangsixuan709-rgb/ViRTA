"""Learning-rate multiplier indexed by zero-based optimizer updates."""
import math


def warmup_cosine_factor(step, total_steps, warmup_steps, start_ratio=0.01, min_ratio=0.01):
    if total_steps < 2 or not 0 <= warmup_steps < total_steps:
        raise ValueError('Require total_steps >= 2 and 0 <= warmup_steps < total_steps')
    if not 0 < start_ratio <= 1 or not 0 < min_ratio <= 1:
        raise ValueError('Learning-rate ratios must be in (0, 1]')
    step = min(max(step, 0), total_steps - 1)
    if warmup_steps and step < warmup_steps:
        return start_ratio + (1 - start_ratio) * step / max(warmup_steps - 1, 1)
    peak_step = max(warmup_steps - 1, 0)
    progress = (step - peak_step) / (total_steps - 1 - peak_step)
    return min_ratio + (1 - min_ratio) * (1 + math.cos(math.pi * progress)) / 2
