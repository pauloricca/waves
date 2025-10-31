from __future__ import annotations
import math
import numpy as np
from typing import Optional
from pydantic import ConfigDict
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.wavable_value import WavableValue
from config import SAMPLE_RATE


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
#
# MIDI Clock Sync:
#   tempo:
#     source: external  # Get BPM from MIDI clock (default: internal)
#     device: korg      # Optional: specify MIDI device key
#     signal:
#       # Your sound definition
class TempoNodeModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    signal: BaseNodeModel  # The signal to render with tempo context
    bpm: Optional[WavableValue] = None  # BPM as WavableValue (scalar, expression, or node)
    source: str = "internal"  # "internal" (use bpm param) or "external" (MIDI clock)
    device: Optional[str] = None  # Optional MIDI device key for external sync
    is_pass_through: bool = True


class TempoNode(BaseNode):
    def __init__(self, model: TempoNodeModel, node_id: str, state=None, do_initialise_state=True):
        from nodes.node_utils.instantiate_node import instantiate_node
        from nodes.wavable_value import WavableValueNode, WavableValueModel
        from nodes.node_utils.midi_utils import MidiInputManager
        
        super().__init__(model, node_id, state, do_initialise_state)
        self.is_stereo = True  # Tempo is a pass-through node, supports stereo
        self.signal_node = self.instantiate_child_node(model.signal, "signal")
        self.source = model.source.lower()
        self.device_key = model.device
        
        # Validate source parameter
        if self.source not in ["internal", "external"]:
            raise ValueError(f"Tempo source must be 'internal' or 'external', got '{self.source}'")
        
        # Wrap bpm in WavableValue if provided
        if model.bpm is not None:
            self.bpm_node = self.instantiate_child_node(model.bpm, "bpm")
        else:
            self.bpm_node = None
        
        # Initialize MIDI manager for external sync
        if self.source == "external":
            self.midi_manager = MidiInputManager()
        
        # Persistent state for hot reload - preserve timing across reloads
        # We override time_since_start to use state instead of instance attribute
        if do_initialise_state:
            self.state.time_since_start = 0
            self.state.number_of_chunks_rendered = 0
    
    def _render_with_timing(self, num_samples: int, context, num_channels: int, **params):
        """Override to use state-based timing instead of instance attributes."""
        from utils import samples_to_time
        
        if self._last_chunk_samples is not None:
            self.state.number_of_chunks_rendered += self._last_chunk_samples
        self.state.time_since_start = samples_to_time(self.state.number_of_chunks_rendered)
        if num_samples is not None:
            self._last_chunk_samples = num_samples
        
        # Call parent's rendering logic (stereo/mono handling)
        return self._do_render(num_samples, context, num_channels, **params)

    
    def _do_render(self, num_samples=None, context=None, num_channels=1, **params):
        # Resolve chunk length in samples
        num_samples_resolved = self.resolve_num_samples(num_samples)
        if num_samples_resolved is None:
            raise ValueError("Tempo node requires explicit duration")

        # Get BPM based on source
        if self.source == "external":
            # Get BPM from MIDI clock
            bpm = self.midi_manager.get_midi_clock_bpm(device_key=self.device_key)

            if bpm is None:
                # No MIDI clock received yet - fall back to bpm parameter if provided
                if self.bpm_node is not None:
                    bpm_wave = self.bpm_node.render(num_samples_resolved, context, **self.get_params_for_children(params))
                    bpm = float(bpm_wave[0]) if len(bpm_wave) > 0 else None
                    if bpm is None:
                        raise ValueError("Tempo node: No MIDI clock available and BPM parameter evaluation returned no value")
                else:
                    raise ValueError("Tempo node: No MIDI clock available and no BPM parameter provided")
        else:
            # Internal source: use bpm parameter
            if self.bpm_node is None:
                raise ValueError("Tempo node with source='internal' requires 'bpm' parameter")

            bpm_wave = self.bpm_node.render(num_samples_resolved, context, **self.get_params_for_children(params))
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
        
        # Generate tick arrays: arrays of length num_samples_resolved with a single 1.0 at the
        # sample corresponding to the start of each bar/beat/eighth/sixteenth/triplet that
        # falls within this chunk. All other samples are 0. These can be used as triggers.
        start_time = self.state.time_since_start  # seconds at the start of this chunk
        end_time = start_time + (num_samples_resolved / SAMPLE_RATE)

        def make_ticks(period_seconds: float) -> np.ndarray:
            ticks = np.zeros(num_samples_resolved, dtype=np.float32)
            if period_seconds <= 0:
                return ticks

            # Determine first and last integer multiples within [start_time, end_time)
            first_n = math.ceil((start_time - 1e-12) / period_seconds)
            last_n = math.floor((end_time - 1e-12) / period_seconds)

            for n in range(first_n, last_n + 1):
                t = n * period_seconds
                idx = int(round((t - start_time) * SAMPLE_RATE))
                if 0 <= idx < num_samples_resolved:
                    ticks[idx] = 1.0
            return ticks

        extended_params['bar_tick'] = make_ticks(bar)
        extended_params['beat_tick'] = make_ticks(beat)
        extended_params['eighth_tick'] = make_ticks(eighth)
        extended_params['sixteenth_tick'] = make_ticks(sixteenth)
        extended_params['triplet_tick'] = make_ticks(triplet)
        
        # Render signal with extended params
        return self.signal_node.render(num_samples, context, num_channels, **extended_params)


TEMPO_DEFINITION = NodeDefinition("tempo", TempoNode, TempoNodeModel)
