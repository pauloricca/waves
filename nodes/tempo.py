from __future__ import annotations
import numpy as np
from typing import Optional
from pydantic import ConfigDict
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.wavable_value import WavableValue


# Tempo node: Adds tempo-related timing variables to render params.
# Provides musical time divisions (bar, beat, eighth, sixteenth, triplet) based on BPM.
# Assumes 4/4 time signature (can be extended for other signatures in the future).
#
# Usage: Wrap your signal with tempo node to access timing variables in expressions:
#   tempo:
#     bpm: 120
#     signal:
#       expression:
#         exp: "sin(t * tau / beat * 4) * 0.5"  # Oscillate 4 times per beat
class TempoNodeModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    signal: BaseNodeModel  # The signal to render with tempo context
    bpm: Optional[WavableValue] = None  # BPM as WavableValue (scalar, expression, or node)
    is_pass_through: bool = True


class TempoNode(BaseNode):
    def __init__(self, model: TempoNodeModel, node_id: str, state=None, do_initialise_state=True):
        from nodes.node_utils.instantiate_node import instantiate_node
        from nodes.wavable_value import WavableValueNode, WavableValueModel
        super().__init__(model, node_id, state, do_initialise_state)
        self.is_stereo = True  # Tempo is a pass-through node, supports stereo
        self.signal_node = self.instantiate_child_node(model.signal, "signal")
        
        # Wrap bpm in WavableValue if provided
        if model.bpm is not None:
            self.bpm_node = self.instantiate_child_node(model.bpm, "bpm")
        else:
            self.bpm_node = None
        
        # TODO: Add MIDI clock support
        # self.midi_clock_source = None
    
    def _do_render(self, num_samples=None, context=None, num_channels=1, **params):
        # Get BPM (from parameter or future MIDI clock)
        if self.bpm_node is None:
            # TODO: Get from MIDI clock when implemented
            # For now, require explicit BPM
            raise ValueError("Tempo node requires 'bpm' parameter (MIDI clock not yet implemented)")
        
        # Render BPM (handles scalars, expressions, and nodes)
        num_samples_resolved = self.resolve_num_samples(num_samples)
        if num_samples_resolved is None:
            raise ValueError("Tempo node requires explicit duration")
        
        bpm_wave = self.bpm_node.render(num_samples_resolved, context, **self.get_params_for_children(params))
        
        # Use first value if BPM is a wave (assume constant BPM for now)
        # TODO: Support time-varying BPM in the future
        bpm = float(bpm_wave[0]) if len(bpm_wave) > 0 else None
        
        if bpm is None:
            raise ValueError("Tempo node: BPM evaluation returned no value")
        
        # Calculate tempo-related values (assuming 4/4 time signature)
        # In 4/4: 1 bar = 4 beats
        beat = 60.0 / bpm  # Length of one beat in seconds
        bar = beat * 4  # Length of one bar in seconds (4 beats in 4/4)
        eighth = beat / 2  # Eighth note
        sixteenth = beat / 4  # Sixteenth note
        triplet = beat / 3  # Eighth note triplet (3 notes per beat)
        
        # Add tempo variables to render params
        extended_params = params.copy()
        extended_params['bpm'] = bpm
        extended_params['bar'] = bar
        extended_params['beat'] = beat
        extended_params['eighth'] = eighth
        extended_params['sixteenth'] = sixteenth
        extended_params['triplet'] = triplet
        
        # Render signal with extended params
        return self.signal_node.render(num_samples, context, num_channels, **extended_params)


TEMPO_DEFINITION = NodeDefinition("tempo", TempoNode, TempoNodeModel)
