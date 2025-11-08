"""
Range Mapping Utilities

Provides unified range mapping functionality for nodes that need to map values
from one range to another. Supports both static scalar ranges and dynamic 
WavableValue ranges for both source and target ranges.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, Optional
import numpy as np

if TYPE_CHECKING:
    from nodes.node_utils.base_node import BaseNode

from nodes.wavable_value import WavableValue


def map_to_range(
    signal: np.ndarray,
    to_min: np.ndarray | float,
    to_max: np.ndarray | float,
    from_min: np.ndarray | float = -1.0,
    from_max: np.ndarray | float = 1.0
) -> np.ndarray:
    """
    Map values from one range to another using vectorized NumPy operations.
    
    This is the core mapping function that handles both static and dynamic ranges.
    It assumes the input signal is in the [from_min, from_max] range and maps it
    to [to_min, to_max]. Both source and target ranges can be dynamic.
    
    Uses NumPy broadcasting for efficiency - keeps scalars as scalars when possible.
    
    Args:
        signal: Input signal array to map
        to_min: Minimum value(s) of output range (scalar or array)
        to_max: Maximum value(s) of output range (scalar or array)
        from_min: Minimum value(s) of input range (default -1.0, can be scalar or array)
        from_max: Maximum value(s) of input range (default 1.0, can be scalar or array)
    
    Returns:
        Mapped signal in the new range
    
    Examples:
        # Map from [-1, 1] to [0, 1]
        mapped = map_to_range(signal, 0.0, 1.0)
        
        # Map from [-1, 1] to [100, 1000] with dynamic max
        mapped = map_to_range(signal, 100.0, lfo_wave)
        
        # Map from [0, 1] to [200, 800]
        mapped = map_to_range(signal, 200, 800, from_min=0, from_max=1)
        
        # Map with dynamic from and to ranges
        mapped = map_to_range(signal, to_min_wave, to_max_wave, from_min_wave, from_max_wave)
    """
    # Handle stereo signal with mono parameters - reshape for broadcasting
    # If signal is 2D (stereo) and parameters are 1D (mono), reshape params to (N, 1)
    signal_is_stereo = signal.ndim == 2
    
    if signal_is_stereo:
        # Reshape 1D parameters to (N, 1) for proper broadcasting with (N, 2) stereo signal
        if isinstance(from_min, np.ndarray) and from_min.ndim == 1:
            from_min = from_min.reshape(-1, 1)
        if isinstance(from_max, np.ndarray) and from_max.ndim == 1:
            from_max = from_max.reshape(-1, 1)
        if isinstance(to_min, np.ndarray) and to_min.ndim == 1:
            to_min = to_min.reshape(-1, 1)
        if isinstance(to_max, np.ndarray) and to_max.ndim == 1:
            to_max = to_max.reshape(-1, 1)
    
    # Calculate ranges - NumPy broadcasting handles scalar/array mixing automatically
    from_range = from_max - from_min
    
    # Check if we have any arrays (dynamic ranges)
    has_dynamic_from = isinstance(from_min, np.ndarray) or isinstance(from_max, np.ndarray)
    
    # Normalize from source range to [0, 1]
    if has_dynamic_from:
        # Handle potential division by zero with dynamic ranges
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized = np.where(from_range != 0, 
                                 (signal - from_min) / from_range,
                                 0.5)
    else:
        # Scalar from_range - simpler and faster path
        if from_range != 0:
            normalized = (signal - from_min) / from_range
        else:
            normalized = np.full_like(signal, 0.5)
    
    # Map to target range - broadcasting handles scalar/array mixing
    to_range = to_max - to_min
    return to_min + normalized * to_range


class RangeMapper:
    """
    Helper class for nodes that need range mapping with WavableValue support.
    
    This class handles the instantiation of child nodes for both source (from) 
    and target (to) range parameters, and provides a render method that applies 
    the mapping. Both ranges support WavableValues.
    
    Usage in a node:
        class MyNode(BaseNode):
            def __init__(self, model, node_id, state, do_initialise_state=True):
                super().__init__(model, node_id, state, do_initialise_state)
                # ... other initialization
                
                # Create range mapper with default from range
                self.range_mapper = RangeMapper.from_model_range(
                    self, model.range, "range"
                ) if model.range else None
            
            def _do_render(self, num_samples=None, context=None, num_channels=1, **params):
                # ... render signal
                
                # Apply range mapping if specified
                if self.range_mapper:
                    signal = self.range_mapper.map(
                        signal, num_samples, context, **params
                    )
                
                return signal
    """
    
    def __init__(
        self, 
        parent_node: BaseNode,
        to_min: WavableValue,
        to_max: WavableValue,
        from_min: WavableValue = -1.0,
        from_max: WavableValue = 1.0,
        to_attribute_name: str = "to",
        from_attribute_name: str = "from"
    ):
        """
        Initialize a RangeMapper with both source and target ranges.
        
        Args:
            parent_node: The node that owns this mapper
            to_min: WavableValue for the minimum of the target range
            to_max: WavableValue for the maximum of the target range
            from_min: WavableValue for the minimum of the source range (default -1.0)
            from_max: WavableValue for the maximum of the source range (default 1.0)
            to_attribute_name: Base name for the target range child node attributes
            from_attribute_name: Base name for the source range child node attributes
        """
        self.parent_node = parent_node
        
        # Target range (to)
        self.to_min_node = parent_node.instantiate_child_node(
            to_min, f"{to_attribute_name}_min"
        )
        self.to_max_node = parent_node.instantiate_child_node(
            to_max, f"{to_attribute_name}_max"
        )
        
        # Source range (from)
        self.from_min_node = parent_node.instantiate_child_node(
            from_min, f"{from_attribute_name}_min"
        )
        self.from_max_node = parent_node.instantiate_child_node(
            from_max, f"{from_attribute_name}_max"
        )
    
    @classmethod
    def from_model_range(
        cls,
        parent_node: BaseNode,
        to_range: Optional[Tuple[WavableValue, WavableValue]],
        to_attribute_name: str = "to",
        from_range: Optional[Tuple[WavableValue, WavableValue]] = None,
        from_attribute_name: str = "from"
    ) -> Optional['RangeMapper']:
        """
        Create a RangeMapper from model range parameters.
        
        Args:
            parent_node: The node that owns this mapper
            to_range: Tuple of (min, max) WavableValues for target range, or None
            to_attribute_name: Base name for the target range child node attributes
            from_range: Tuple of (min, max) WavableValues for source range, or None (defaults to [-1, 1])
            from_attribute_name: Base name for the source range child node attributes
        
        Returns:
            RangeMapper instance if to_range is not None, else None
        """
        if to_range is None:
            return None
        
        # Default from range is [-1, 1] if not specified
        if from_range is None:
            from_min, from_max = -1.0, 1.0
        else:
            from_min, from_max = from_range[0], from_range[1]
        
        return cls(
            parent_node, 
            to_range[0], to_range[1],
            from_min, from_max,
            to_attribute_name, from_attribute_name
        )
    
    def map(
        self,
        signal: np.ndarray,
        num_samples: int,
        context,
        **params
    ) -> np.ndarray:
        """
        Apply range mapping to a signal.
        
        Args:
            signal: Input signal array to map
            num_samples: Number of samples to render for range parameters
            context: Render context
            **params: Additional render parameters to pass to child nodes
        
        Returns:
            Mapped signal in the new range
        """
        # Use actual signal length, not requested num_samples
        # (signal might be shorter if it's ending)
        actual_num_samples = len(signal)
        
        # Render all range parameters
        child_params = self.parent_node.get_params_for_children(params)
        to_min = self.to_min_node.render(actual_num_samples, context, **child_params)
        to_max = self.to_max_node.render(actual_num_samples, context, **child_params)
        from_min = self.from_min_node.render(actual_num_samples, context, **child_params)
        from_max = self.from_max_node.render(actual_num_samples, context, **child_params)
        
        # Apply mapping
        return map_to_range(signal, to_min, to_max, from_min, from_max)
