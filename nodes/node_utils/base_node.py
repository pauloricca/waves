
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


    def render(self, num_samples: int, **kwargs) -> np.ndarray:
        """
        Renders the node for a given number of samples.
        Any kwargs passed to this function should be forwarded to the node's children,
        The node can use consume_kwargs to consume and remove some of the kwargs, if needed,
        or get_kwargs_for_children othewise before passing them on to the children, 
        e.g. an oscillator node might want to consume frequency multiplier and amplitude
        multiplier kwargs but we don't want to pass them to the child oscillator nodes.
        """
        self.number_of_chunks_rendered += self._last_chunk_samples
        self.time_since_start = self.number_of_chunks_rendered / SAMPLE_RATE
        self._last_chunk_samples = num_samples
    

    def get_kwargs_for_children(self, kwargs: dict, args_to_remove: list[str] = []) -> dict:
        """
        Returns a new kwargs dict with the specified args removed.
        This is useful for passing kwargs to child nodes without including
        certain args that are not relevant to them.
        """
        new_kwargs = kwargs.copy()
        for arg in args_to_remove:
            new_kwargs.pop(arg, None)
        
        if self.duration is not None:
            new_kwargs[RenderArgs.DURATION] = self.duration

        return new_kwargs


    def consume_kwargs(self, kwargs: dict, keys_and_default_values: dict):
        """
        Extracts keys from `kwargs` using defaults from `defaults`, without mutating the original.
        Returns a tuple of values (in the order of keys_and_default_values) and a new kwargs dict
        with those keys removed.
        """
        new_kwargs = kwargs.copy()
        values = []
        for key, default in keys_and_default_values.items():
            values.append(new_kwargs.pop(key, default))

        if self.duration is not None:
            new_kwargs[RenderArgs.DURATION] = self.duration

        return (*values, new_kwargs)