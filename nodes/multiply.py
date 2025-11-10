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
    signal: WavableValue  # The signal to multiply


class MultiplyNode(BaseNode):
    def __init__(self, model: MultiplyNodeModel, node_id: str, state=None, do_initialise_state=True):
        super().__init__(model, node_id, state, do_initialise_state)
        self.model = model
        
        # Validate that either number or items is provided, but not both
        if model.number is None and model.items is None:
            raise ValueError("Multiply node requires either 'number' or 'items' parameter")
        if model.number is not None and model.items is not None:
            raise ValueError("Multiply node cannot have both 'number' and 'items' parameters")
        
        self.number_node = None
        if model.number is not None:
            self.number_node = self.instantiate_child_node(model.number, "number")
        
        # Persistent state: track how many instances we've created
        if do_initialise_state:
            self.state.instance_count = 0
        
        # Ephemeral: Recreate signal instances from state (for hot reload)
        self.signal_instances: list[BaseNode] = []
        for i in range(self.state.instance_count):
            signal_copy = self.instantiate_child_node(self.model.signal, "signal", i)
            self.signal_instances.append(signal_copy)
    
    def _do_render(self, num_samples=None, context=None, **params):
        num_samples = self.resolve_num_samples(num_samples)
        if num_samples is None:
            raise ValueError("Multiply node requires explicit duration")
        
        # Determine count based on whether we're using number or items
        if self.number_node is not None:
            # Number mode: evaluate the number parameter
            number_wave = self.number_node.render(num_samples, context, **self.get_params_for_children(params))
            count = int(number_wave[0])
            items_mode = False
        else:
            # Items mode: count is the length of the items list
            count = len(self.model.items)
            items_mode = True
        
        if len(self.signal_instances) < count:
            new_instances: list[BaseNode] = []
            for i_diff in range(count - len(self.signal_instances)):
                i = len(self.signal_instances) + i_diff
                # Create a copy of the signal model with a stable ID
                signal_copy = self.instantiate_child_node(self.model.signal, "signal", i)
                new_instances.append(signal_copy)
            self.signal_instances.extend(new_instances)
            # Update state to track instance count
            self.state.instance_count = len(self.signal_instances)
        
        elif len(self.signal_instances) > count:
            self.signal_instances = self.signal_instances[:count]
            # Update state to track instance count
            self.state.instance_count = len(self.signal_instances)

        if count <= 0:
            return np.zeros(num_samples, dtype=np.float32)
        
        # Render all instances with extended context params
        mixed_wave = None
        
        for i, signal_node in enumerate(self.signal_instances):
            # Add iteration context variables
            extended_params = params.copy()
            extended_params['i'] = float(i + 1)  # 1-indexed iteration
            extended_params['n'] = float(count)  # Total number of copies
            
            # Add item value if in items mode
            if items_mode:
                extended_params['item'] = self.model.items[i]
            
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
