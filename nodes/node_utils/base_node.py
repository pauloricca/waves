
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


    def render(self, num_samples: int = None, **params) -> np.ndarray:
        """
        Renders the node for a given number of samples.
        
        Args:
            num_samples: Number of samples to render. If None, render the entire duration
                        of the node (based on self.duration or until the node is exhausted).
                        This enables efficient full-signal rendering for nodes like shuffle
                        that need to process the complete signal at once.
        
        Any params passed to this function should be forwarded to the node's children,
        The node can use consume_params to consume and remove some of the params, if needed,
        or get_params_for_children othewise before passing them on to the children, 
        e.g. an oscillator node might want to consume frequency multiplier and amplitude
        multiplier params but we don't want to pass them to the child oscillator nodes.
        """
        self.number_of_chunks_rendered += self._last_chunk_samples
        self.time_since_start = self.number_of_chunks_rendered / SAMPLE_RATE
        if num_samples is not None:
            self._last_chunk_samples = num_samples
        # If num_samples is None, _last_chunk_samples will be updated by the implementing node
    

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


    def resolve_num_samples(self, num_samples: Optional[int]) -> Optional[int]:
        """
        Helper method to resolve num_samples when it's None.
        If num_samples is None and duration is set, calculates num_samples from duration.
        Otherwise returns None (caller should handle getting full child signal).
        
        Args:
            num_samples: The requested number of samples, or None
            
        Returns:
            Resolved num_samples, or None if it should be determined from child
        """
        if num_samples is None and self.duration is not None:
            num_samples = int(self.duration * SAMPLE_RATE)
            self._last_chunk_samples = num_samples
        return num_samples
    
    
    def render_full_child_signal(self, child_node, **params) -> np.ndarray:
        """
        Helper method to render the full signal from a child node.
        This is useful when the node needs the complete child signal to process
        (e.g., for normalization, filtering, shuffling).
        
        Args:
            child_node: The child node to render
            **params: Parameters to pass to the child
            
        Returns:
            The full signal from the child node
        """
        signal = child_node.render(**params)
        if len(signal) > 0:
            self._last_chunk_samples = len(signal)
        return signal
