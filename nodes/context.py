from __future__ import annotations
from pydantic import ConfigDict
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.wavable_value import WavableValue
from utils import empty_mono, is_stereo, to_mono


# Context node: Creates a scope where additional variables are added to render params.
# Takes a signal plus any number of named WavableValue arguments. These arguments are
# rendered first and added as variables to the render params before rendering the signal.
class ContextNodeModel(BaseNodeModel):
    model_config = ConfigDict(extra='allow')  # Allow arbitrary named parameters
    signal: WavableValue
    is_pass_through: bool = True


class ContextNode(BaseNode):
    def __init__(self, model: ContextNodeModel, node_id: str, state, do_initialise_state=True):
        super().__init__(model, node_id, state, do_initialise_state)
        self.model = model
        
        # Instantiate the signal node
        self.signal_node = self.instantiate_child_node(model.signal, "signal")
        
        # Store all extra arguments as WavableValue nodes
        self.context_args = {}
        if hasattr(model, '__pydantic_extra__') and model.__pydantic_extra__:
            for field_name, field_value in model.__pydantic_extra__.items():
                self.context_args[field_name] = self.instantiate_child_node(field_value, field_name)
        
        # Track total samples rendered for duration checking
        if do_initialise_state:
            self.state.total_samples_rendered = 0
    
    def _do_render(self, num_samples=None, context=None, **params):
        # Resolve num_samples first (handles None case)
        num_samples = self.resolve_num_samples(num_samples)
        if num_samples is None:
            raise ValueError("Context node requires explicit duration")
        
        # Check if we've already rendered the full duration
        if self.duration is not None:
            from config import SAMPLE_RATE
            max_samples = int(self.duration * SAMPLE_RATE)
            
            # If we've already rendered everything, return empty
            if self.state.total_samples_rendered >= max_samples:
                return empty_mono()
            
            # If num_samples would exceed duration, limit it
            remaining_samples = max_samples - self.state.total_samples_rendered
            if remaining_samples <= 0:
                return empty_mono()
            num_samples = min(num_samples, remaining_samples)
        
        # Render all context arguments and add to params
        # Each variable can reference previously defined ones
        extended_params = params.copy()
        
        for name, node in self.context_args.items():
            # Render context args with accumulated params (may be mono or stereo)
            wave = node.render(num_samples, context, **self.get_params_for_children(extended_params))
            # If child returned empty, we're done
            if len(wave) == 0:
                return empty_mono()
            # Convert stereo context args to mono (context variables should be control signals)
            if is_stereo(wave):
                wave = to_mono(wave)
            # Add this variable to extended_params so subsequent variables can use it
            extended_params[name] = wave
        
        # Now render the signal with the extended params (pass through whatever it returns)
        result = self.signal_node.render(num_samples, context, **extended_params)
        
        # Track samples rendered
        self.state.total_samples_rendered += len(result)
        
        return result


CONTEXT_DEFINITION = NodeDefinition("context", ContextNode, ContextNodeModel)
