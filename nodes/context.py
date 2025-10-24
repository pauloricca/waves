from __future__ import annotations
import numpy as np
from pydantic import ConfigDict
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.wavable_value import WavableValue


# Context node: Creates a scope where additional variables are added to render params.
# Takes a signal plus any number of named WavableValue arguments. These arguments are
# rendered first and added as variables to the render params before rendering the signal.
class ContextNodeModel(BaseNodeModel):
    model_config = ConfigDict(extra='allow')  # Allow arbitrary named parameters
    signal: BaseNodeModel  # The signal to render with the extended context


class ContextNode(BaseNode):
    def __init__(self, model: ContextNodeModel, state, hot_reload=False):
        from nodes.node_utils.instantiate_node import instantiate_node
        from nodes.wavable_value import WavableValueNode, WavableValueModel
        super().__init__(model, state, hot_reload)
        self.model = model
        
        # Instantiate the signal node
        self.signal_node = instantiate_node(model.signal, hot_reload=hot_reload)
        
        # Store all extra arguments as WavableValue nodes
        self.context_args = {}
        if hasattr(model, '__pydantic_extra__') and model.__pydantic_extra__:
            for field_name, field_value in model.__pydantic_extra__.items():
                if isinstance(field_value, BaseNodeModel):
                    # Already a node, instantiate it
                    self.context_args[field_name] = instantiate_node(field_value, hot_reload=hot_reload)
                else:
                    # Wrap in WavableValue (handles scalars, expressions, lists)
                    wavable_model = WavableValueModel(value=field_value)
                    self.context_args[field_name] = WavableValueNode(wavable_model)
    
    def _do_render(self, num_samples=None, context=None, **params):
        num_samples = self.resolve_num_samples(num_samples)
        if num_samples is None:
            raise ValueError("Context node requires explicit duration")
        
        # Render all context arguments and add to params
        extended_params = params.copy()
        
        for name, node in self.context_args.items():
            wave = node.render(num_samples, context, **self.get_params_for_children(params))
            extended_params[name] = wave
        
        # Now render the signal with the extended params
        return self.signal_node.render(num_samples, context, **extended_params)


CONTEXT_DEFINITION = NodeDefinition("context", ContextNode, ContextNodeModel)
