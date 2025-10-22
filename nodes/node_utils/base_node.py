
from __future__ import annotations

import numpy as np

from config import *
from typing import Optional
from pydantic import BaseModel

from constants import RenderArgs


class BaseNodeModel(BaseModel):
    duration: Optional[float] = None
    id: Optional[str] = None  # Unique identifier for this node to enable referencing
    pass


class BaseNode:
    def __init__(self, model: BaseNodeModel):
        self.duration = model.duration
        self.time_since_start = 0
        self.number_of_chunks_rendered = 0
        self._last_chunk_samples = 0
        
        # Get the effective ID (explicit or auto-generated)
        from nodes.node_utils.auto_id_generator import AutoIDGenerator
        self.node_id = AutoIDGenerator.get_effective_id(model)  # Store node id for caching


    def render(self, num_samples: int = None, context = None, **params) -> np.ndarray:
        """
        Main render method that handles caching, recursion tracking, and timing.
        Delegates actual rendering to _do_render() which subclasses implement.
        
        Args:
            num_samples: Number of samples to render. If None, render the entire duration
                        of the node (based on self.duration or until the node is exhausted).
            context: RenderContext for managing shared state, caching, and recursion tracking.
            **params: Parameters to forward to child nodes.
        
        This method:
        1. Creates a default context if none provided (backwards compatibility)
        2. Handles caching and recursion for nodes with an id
        3. Updates timing information
        4. Calls _do_render() for the actual rendering logic
        """
        # Create default context if none provided
        if context is None:
            from nodes.node_utils.render_context import RenderContext
            context = RenderContext()
        
        # If this node has an id, handle caching and recursion
        if self.node_id:
            # Register this node instance in the context so it can be referenced
            context.store_node(self.node_id, self)
            
            # Use Python's id() to track this specific node instance
            instance_id = id(self)
            recursion_depth = context.get_recursion_depth(instance_id)
            
            # Check if we've hit max recursion (feedback loop break)
            if recursion_depth >= context.max_recursion:
                # Return zeros to break feedback loop
                if num_samples is None:
                    return np.array([], dtype=np.float32)
                return np.zeros(num_samples, dtype=np.float32)
            
            # If not in recursion and THIS INSTANCE has a cached output, use it
            # This allows the same node instance to be called multiple times (fan-out)
            # without re-rendering, but different instances with the same ID will render independently
            if recursion_depth == 0:
                cached = context.get_output(instance_id)
                if cached is not None:
                    # Return cached output (trim/pad as needed)
                    return self._adjust_output_length(cached.copy(), num_samples)
            
            # Increment recursion, render, decrement, cache (if top level)
            context.increment_recursion(instance_id)
            try:
                wave = self._render_with_timing(num_samples, context, **params)
                if recursion_depth == 0:  # Only cache at top level
                    context.store_output(instance_id, self.node_id, wave)
                return wave
            finally:
                context.decrement_recursion(instance_id)
        
        # No id, just render normally with timing
        return self._render_with_timing(num_samples, context, **params)
    
    
    def _render_with_timing(self, num_samples: int, context, **params) -> np.ndarray:
        """
        Updates timing info and calls _do_render().
        This keeps timing logic separate from rendering logic.
        """
        if self._last_chunk_samples is not None:
            self.number_of_chunks_rendered += self._last_chunk_samples
        self.time_since_start = self.number_of_chunks_rendered / SAMPLE_RATE
        if num_samples is not None:
            self._last_chunk_samples = num_samples
        # If num_samples is None, _last_chunk_samples will be updated by the implementing node
        
        return self._do_render(num_samples, context, **params)
    
    
    def _do_render(self, num_samples: int, context, **params) -> np.ndarray:
        """
        Subclasses override this to implement their rendering logic.
        This method focuses purely on rendering without worrying about caching,
        recursion, or timing - those are handled by render().
        
        Args:
            num_samples: Number of samples to render
            context: RenderContext for accessing cached outputs and passing to children
            **params: Parameters forwarded from parent nodes
            
        Returns:
            Rendered wave as numpy array
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement _do_render()")
    
    
    def _adjust_output_length(self, wave: np.ndarray, num_samples: Optional[int]) -> np.ndarray:
        """
        Helper to adjust wave length to match requested num_samples.
        Used when returning cached outputs.
        """
        if num_samples is None or len(wave) == 0:
            return wave
        
        if len(wave) < num_samples:
            # Pad with zeros
            padding = np.zeros(num_samples - len(wave), dtype=np.float32)
            return np.concatenate([wave, padding])
        elif len(wave) > num_samples:
            return wave[:num_samples]
        
        return wave
    

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
    
    
    def render_full_child_signal(self, child_node, context=None, **params) -> np.ndarray:
        """
        Helper method to render the full signal from a child node.
        This is useful when the node needs the complete child signal to process
        (e.g., for normalization, filtering, shuffling).
        
        Args:
            child_node: The child node to render
            context: RenderContext to pass to child
            **params: Parameters to pass to the child
            
        Returns:
            The full signal from the child node
        """
        signal = child_node.render(context=context, **params)
        if len(signal) > 0:
            self._last_chunk_samples = len(signal)
        return signal
    
    
    def eval_scalar(self, value, context, **render_params):
        """
        Evaluate a scalar parameter that might be a number or expression string.
        Always returns a single float value (not an array).
        
        Args:
            value: The value to evaluate (number or expression string)
            context: RenderContext (not used currently but kept for consistency)
            **render_params: Render parameters to make available in expressions
            
        Returns:
            A float value
        """
        if isinstance(value, str):
            # It's an expression
            from expression_globals import get_expression_context
            eval_context = get_expression_context(render_params, self.time_since_start, 1)
            
            # Evaluate
            try:
                compiled = compile(value, '<expression>', 'eval')
                result = eval(compiled, {"__builtins__": {}}, eval_context)
                
                # Extract scalar from result
                if isinstance(result, np.ndarray):
                    return float(result[0] if len(result) > 0 else 0)
                else:
                    return float(result)
            except Exception as e:
                raise ValueError(f"Error evaluating scalar expression '{value}': {e}")
        else:
            # Already a scalar
            return float(value)
