
from __future__ import annotations

import numpy as np

from config import *
from typing import Optional
from pydantic import BaseModel

from constants import RenderArgs


class BaseNodeModel(BaseModel):
    duration: Optional[float] = None
    pass


class BaseNode:
    def __init__(self, model: BaseNodeModel):
        self.duration = model.duration
        self.time_since_start = 0
        self.number_of_chunks_rendered = 0
        self._last_chunk_samples = 0


    def render(self, num_samples: int, **params) -> np.ndarray:
        """
        Renders the node for a given number of samples.
        Any params passed to this function should be forwarded to the node's children,
        The node can use consume_params to consume and remove some of the params, if needed,
        or get_params_for_children othewise before passing them on to the children, 
        e.g. an oscillator node might want to consume frequency multiplier and amplitude
        multiplier params but we don't want to pass them to the child oscillator nodes.
        """
        self.number_of_chunks_rendered += self._last_chunk_samples
        self.time_since_start = self.number_of_chunks_rendered / SAMPLE_RATE
        self._last_chunk_samples = num_samples
    

    def get_params_for_children(self, params: dict, args_to_remove: list[str] = []) -> dict:
        """
        Returns a new params dict with the specified args removed.
        This is useful for passing params to child nodes without including
        certain args that are not relevant to them.
        """
        new_params = params.copy()
        for arg in args_to_remove:
            new_params.pop(arg, None)
        
        if self.duration is not None:
            new_params[RenderArgs.DURATION] = self.duration

        return new_params


    def consume_params(self, params: dict, keys_and_default_values: dict):
        """
        Extracts keys from `params` using defaults from `defaults`, without mutating the original.
        Returns a tuple of values (in the order of keys_and_default_values) and a new params dict
        with those keys removed.
        """
        new_params = params.copy()
        values = []
        for key, default in keys_and_default_values.items():
            values.append(new_params.pop(key, default))

        if self.duration is not None:
            new_params[RenderArgs.DURATION] = self.duration

        return (*values, new_params)