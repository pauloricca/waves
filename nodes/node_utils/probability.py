from __future__ import annotations

from random import random
from typing import Any

import numpy as np


def value_to_probability(value: Any) -> float:
    """Normalize bool/numeric/expression results into a probability."""
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return 0.0
        value = value.flat[0]

    if isinstance(value, (bool, np.bool_)):
        return 1.0 if value else 0.0

    return float(np.clip(float(value), 0.0, 1.0))


def evaluate_probability(prob_value, render_params: dict, time: float, num_samples: int, context=None) -> float:
    """Evaluate a probability value against the current render context."""
    if prob_value is None:
        return 1.0

    if isinstance(prob_value, str):
        from expression_globals import evaluate_expression, get_expression_context

        expression_context = get_expression_context(render_params, time, num_samples, context)
        prob_value = evaluate_expression(prob_value, expression_context, num_samples=None)

    return value_to_probability(prob_value)


def should_play_probability(prob_value, render_params: dict, time: float, num_samples: int, context=None) -> bool:
    """Return whether a probabilistic step should play."""
    probability = evaluate_probability(prob_value, render_params, time, num_samples, context)
    return random() < probability
