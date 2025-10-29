from __future__ import annotations
from typing import Optional, List, Union
import numpy as np
from pydantic import ConfigDict
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.wavable_value import WavableValue


# Interval node: Maps note indices to frequencies based on a scale.
# 
# Takes a root frequency and a note index (distance from root in the scale),
# and outputs the corresponding frequency using the specified scale.
#
# Parameters:
# - root: Root note frequency (WavableValue, default: 440 Hz = A4)
# - note: Note index in the scale (WavableValue, 0 = root, 1 = second degree, etc.)
# - scale: Scale definition as semitone intervals from root (list or expression that returns a list)
#          Default: chromatic scale [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
#          Can be a direct list or an expression like 'blues' or 'major'
#
# The note index wraps around the scale (octaves are handled automatically).
# For example, with a 7-note scale, note index 7 = root + 1 octave, 
# index -1 = last note of previous octave.
#
# Examples:
#
# 1. Simple chromatic scale:
#    interval:
#      root: 440
#      note: 7  # 7 semitones up = perfect fifth
#
# 2. Using blues scale with expression:
#    interval:
#      root: 261.63  # C4
#      note:
#        sequencer:
#          pattern: [0, 3, 5, 7, 10, 12]  # Note indices
#      scale: blues  # Expression evaluates to [0, 3, 5, 6, 7, 10]
#
# 3. Switching scales with select:
#    interval:
#      root: 440
#      note: 
#        sequencer:
#          pattern: [0, 2, 4, 7]
#      scale:
#        select:
#          test: "t < 2"
#          "true": major
#          "false": minor_pentatonic
#
# 4. Dynamic root with LFO modulation:
#    interval:
#      root:
#        osc:
#          type: sin
#          freq: 0.5
#          range: [220, 440]
#      note: 4
#      scale: major


class IntervalModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    root: WavableValue = 440.0  # A4
    note: WavableValue = 0  # Root note by default
    scale: Optional[Union[List[int], str, BaseNodeModel]] = None  # Scale definition or expression


class IntervalNode(BaseNode):
    def __init__(self, model: IntervalModel, node_id: str, state=None, do_initialise_state=True):
        from expression_globals import compile_expression
        super().__init__(model, node_id, state, do_initialise_state)
        
        # Instantiate root and note as child nodes
        self.root_node = self.instantiate_child_node(model.root, "root")
        self.note_node = self.instantiate_child_node(model.note, "note")
        
        # Handle scale parameter
        if model.scale is None:
            # Default chromatic scale
            self.scale_type = 'constant'
            self.scale_value = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.float32)
            self.scale_node = None
        elif isinstance(model.scale, list):
            # Direct list
            self.scale_type = 'constant'
            self.scale_value = np.array(model.scale, dtype=np.float32)
            self.scale_node = None
        elif isinstance(model.scale, str):
            # Expression that should evaluate to a list
            self.scale_type = 'expression'
            self.compiled_scale = compile_expression(model.scale)
            self.scale_node = None
        elif isinstance(model.scale, BaseNodeModel):
            # Node that outputs a scale (e.g., select node)
            self.scale_type = 'node'
            self.scale_node = self.instantiate_child_node(model.scale, "scale")
        else:
            raise ValueError(f"Invalid scale type: {type(model.scale)}")
    
    def _get_scale(self, context, **params):
        """Get the current scale as a numpy array."""
        from expression_globals import evaluate_compiled, get_expression_context
        
        if self.scale_type == 'constant':
            return self.scale_value
        elif self.scale_type == 'expression':
            # Evaluate expression to get scale
            eval_context = get_expression_context(params, self.time_since_start, 1, context)
            result = evaluate_compiled(self.compiled_scale, eval_context, num_samples=None)
            
            # Result should be a list or array
            if isinstance(result, (list, np.ndarray)):
                return np.array(result, dtype=np.float32)
            else:
                raise ValueError(f"Scale expression must evaluate to a list, got {type(result)}")
        elif self.scale_type == 'node':
            # This is for future use - nodes that output changing scales
            # For now, we just render one sample and expect a constant value
            # In practice, this would likely be a select node that returns different scale constants
            scale_output = self.scale_node.render(1, context, **self.get_params_for_children(params))
            # This is a bit of a hack - we assume the node will output something we can interpret
            # In practice, select nodes with expression outputs would work
            raise NotImplementedError("Scale as node not yet fully implemented - use expressions or select with scale names")
        
        return self.scale_value
    
    def _note_to_frequency(self, root_freq, note_index, scale):
        """
        Convert a note index to a frequency based on the scale.
        
        Args:
            root_freq: Root frequency (can be scalar or array)
            note_index: Note index in the scale (can be scalar or array)
            scale: Scale as array of semitone intervals
        
        Returns:
            Frequency (scalar or array matching input shapes)
        """
        # Ensure inputs are arrays for vectorization
        root_freq = np.atleast_1d(root_freq)
        note_index = np.atleast_1d(note_index)
        
        # Round note indices to integers
        note_index = np.round(note_index).astype(int)
        
        # Calculate octave offset and position within scale
        scale_len = len(scale)
        octave = note_index // scale_len
        position = note_index % scale_len
        
        # Get semitone offset from scale
        # Handle negative indices properly
        semitone_offset = scale[position] + octave * 12
        
        # Calculate frequency using equal temperament
        # freq = root * 2^(semitones/12)
        frequency = root_freq * np.power(2.0, semitone_offset / 12.0)
        
        return frequency.astype(np.float32)
    
    def _do_render(self, num_samples=None, context=None, **params):
        num_samples = self.resolve_num_samples(num_samples)
        if num_samples is None:
            raise ValueError("Interval node requires explicit duration")
        
        # Get the scale for this render
        scale = self._get_scale(context, **params)
        
        # Render root frequency and note index
        root_freq = self.root_node.render(num_samples, context, **self.get_params_for_children(params))
        note_index = self.note_node.render(num_samples, context, **self.get_params_for_children(params))
        
        # Handle early termination
        if len(root_freq) == 0 or len(note_index) == 0:
            return np.array([], dtype=np.float32)
        
        # Take minimum length
        actual_samples = min(len(root_freq), len(note_index))
        if actual_samples < num_samples:
            root_freq = root_freq[:actual_samples]
            note_index = note_index[:actual_samples]
        
        # Convert note indices to frequencies
        output = self._note_to_frequency(root_freq, note_index, scale)
        
        return output[:actual_samples]


INTERVAL_DEFINITION = NodeDefinition(
    name="interval",
    model=IntervalModel,
    node=IntervalNode
)
