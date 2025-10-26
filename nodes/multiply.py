"""
Multiply Node - Creates multiple copies of a signal with iteration context

This node creates N copies of a signal node and mixes them together.
It provides context variables 'i' (1-indexed iteration) and 'n' (total count)
that can be used in expressions to vary parameters across copies.

When 'items' is provided, the node acts like a foreach loop, providing an
additional 'item' context variable containing the value from the items list.

Example usage in YAML:
    chorus_effect:
      multiply:
        number: 5
        signal:
          osc:
            type: sin
            freq: "440 * (1 + (i-1) * 0.01)"  # Slightly detune each copy (i goes 1-5)
            phase: "(i-1) / n"                 # Distribute phases evenly
            amp: "1 / n"                       # Normalize amplitude

    harmonic_stack:
      multiply:
        number: 8
        signal:
          osc:
            type: sin
            freq: "220 * i"        # Create harmonics (220, 440, 660, ...) (i goes 1-8)
            amp: "0.5 / i"         # Reduce amplitude for higher harmonics
    
    foreach_frequencies:
      multiply:
        items: [220, 440, 660, 880]  # Or use a node that outputs values
        signal:
          osc:
            type: sin
            freq: item     # Use the current item value
            amp: "0.25 / n"
"""

from __future__ import annotations
import numpy as np
from typing import Optional, List
from pydantic import ConfigDict
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.wavable_value import WavableValue
from utils import add_waves


class MultiplyNodeModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    number: Optional[WavableValue] = None  # Number of copies to create
    items: Optional[List[WavableValue]] = None  # List of items to iterate over (foreach mode)
    signal: BaseNodeModel  # The signal to multiply


class MultiplyNode(BaseNode):
    def __init__(self, model: MultiplyNodeModel, state=None, hot_reload=False):
        from nodes.node_utils.instantiate_node import instantiate_node
        from nodes.node_utils.auto_id_generator import AutoIDGenerator
        from nodes.wavable_value import WavableValueNode, WavableValueModel
        
        super().__init__(model, state, hot_reload)
        self.model = model
        
        # Get my ID for creating stable child IDs
        my_id = AutoIDGenerator.get_effective_id(model)
        
        # Validate that either number or items is provided, but not both
        if model.number is None and model.items is None:
            raise ValueError("Multiply node requires either 'number' or 'items' parameter")
        if model.number is not None and model.items is not None:
            raise ValueError("Multiply node cannot have both 'number' and 'items' parameters")
        
        # Create WavableValue node for the number parameter (if provided)
        self.number_node = None
        if model.number is not None:
            if isinstance(model.number, BaseNodeModel):
                self.number_node = instantiate_node(model.number, hot_reload=True)
            else:
                wavable_model = WavableValueModel(value=model.number)
                wavable_model.__auto_id__ = f"{my_id}.number"
                self.number_node = WavableValueNode(wavable_model)
        
        # Create WavableValue nodes for items (if provided)
        self.item_nodes = None
        if model.items is not None:
            self.item_nodes = []
            for idx, item_value in enumerate(model.items):
                if isinstance(item_value, BaseNodeModel):
                    self.item_nodes.append(instantiate_node(item_value, hot_reload=True))
                else:
                    wavable_model = WavableValueModel(value=item_value)
                    wavable_model.__auto_id__ = f"{my_id}.items.{idx}"
                    self.item_nodes.append(WavableValueNode(wavable_model))
        
        # Store the signal model - we'll instantiate copies dynamically
        self.signal_model = model.signal
        
        # Cache for signal instances - keyed by count to handle dynamic number changes
        self.signal_instances = {}
    
    def _do_render(self, num_samples=None, context=None, **params):
        from nodes.node_utils.instantiate_node import instantiate_node
        from nodes.node_utils.auto_id_generator import AutoIDGenerator
        
        num_samples = self.resolve_num_samples(num_samples)
        if num_samples is None:
            raise ValueError("Multiply node requires explicit duration")
        
        # Determine count based on whether we're using number or items
        if self.number_node is not None:
            # Number mode: evaluate the number parameter
            number_wave = self.number_node.render(num_samples, context, **self.get_params_for_children(params))
            count = int(number_wave[0]) if isinstance(number_wave, np.ndarray) else int(number_wave)
            items_mode = False
            item_values = None
        else:
            # Items mode: count is the length of the items list
            count = len(self.item_nodes)
            items_mode = True
            # Render all item values
            item_values = []
            for item_node in self.item_nodes:
                item_wave = item_node.render(num_samples, context, **self.get_params_for_children(params))
                item_values.append(item_wave)
        
        if count <= 0:
            return np.zeros(num_samples, dtype=np.float32)
        
        # Get or create signal instances for this count
        # For items mode, we use a special key to distinguish it from number mode
        cache_key = f"items_{count}" if items_mode else count
        
        if cache_key not in self.signal_instances:
            self.signal_instances[cache_key] = []
            my_id = AutoIDGenerator.get_effective_id(self.model)
            
            for i in range(count):
                # Create a copy of the signal model with a stable ID
                signal_copy = instantiate_node(self.signal_model, hot_reload=True)
                
                # Assign a stable auto-ID for this specific instance
                if hasattr(signal_copy, 'model'):
                    if items_mode:
                        signal_copy.model.__auto_id__ = f"{my_id}.signal.items.{i}"
                    else:
                        signal_copy.model.__auto_id__ = f"{my_id}.signal.{count}.{i}"
                
                self.signal_instances[cache_key].append(signal_copy)
        
        # Render all instances with extended context params
        mixed_wave = None
        
        for i, signal_node in enumerate(self.signal_instances[cache_key]):
            # Add iteration context variables
            extended_params = params.copy()
            extended_params['i'] = float(i + 1)  # 1-indexed iteration
            extended_params['n'] = float(count)  # Total number of copies
            
            # Add item value if in items mode
            if items_mode:
                extended_params['item'] = item_values[i]
            
            # Render this instance
            signal_wave = signal_node.render(num_samples, context, **extended_params)
            
            if len(signal_wave) > 0:
                if mixed_wave is None:
                    mixed_wave = signal_wave.copy()
                else:
                    mixed_wave = add_waves(mixed_wave, signal_wave)
        
        # Return the mixed result or silence
        if mixed_wave is None:
            return np.zeros(num_samples, dtype=np.float32)
        
        return mixed_wave


MULTIPLY_DEFINITION = NodeDefinition("multiply", MultiplyNode, MultiplyNodeModel)
