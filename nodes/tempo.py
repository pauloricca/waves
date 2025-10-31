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
# Exposes the following parameters to child nodes:
# - bpm: Current BPM (beats per minute)
# - bar, beat, eighth, sixteenth, triplet: Duration of each division in seconds
# - bar_number, beat_number, eighth_number, sixteenth_number, triplet_number: 
#   Integer count of current bar/beat/etc (1-indexed)
# - bar_tick, beat_tick, eighth_tick, sixteenth_tick, triplet_tick:
#   Arrays with 1.0 at division boundaries, 0 elsewhere (trigger signals)
# - two_bar_tick, four_bar_tick, eight_bar_tick, sixteen_bar_tick:
#   Arrays with 1.0 at multi-bar boundaries (2/4/8/16 bars)
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
        from nodes.node_utils.midi_utils import MidiInputManager, MidiOutputManager
        
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
            self.midi_input_manager = MidiInputManager()
        
        # Initialize MIDI output manager for internal clock output
        if self.source == "internal":
            self.midi_output_manager = MidiOutputManager()
        
        # Persistent state for hot reload - preserve timing across reloads
        # We override time_since_start to use state instead of instance attribute
        if do_initialise_state:
            self.state.time_since_start = 0
            self.state.number_of_chunks_rendered = 0
            self.state.beats_since_start = 0.0  # Track accumulated beats for dynamic BPM
            self.state.last_bpm = None  # Track last BPM for MIDI clock updates
    
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
            bpm = self.midi_input_manager.get_midi_clock_bpm(device_key=self.device_key)

            if bpm is None:
                # No MIDI clock received yet - fall back to bpm parameter if provided
                if self.bpm_node is not None:
                    bpm_wave = self.bpm_node.render(num_samples_resolved, context, **self.get_params_for_children(params))
                    if len(bpm_wave) == 0:
                        raise ValueError("Tempo node: No MIDI clock available and BPM parameter evaluation returned no value")
                    bpm = float(np.mean(bpm_wave))
                else:
                    raise ValueError("Tempo node: No MIDI clock available and no BPM parameter provided")
        else:
            # Internal source: use bpm parameter
            if self.bpm_node is None:
                raise ValueError("Tempo node with source='internal' requires 'bpm' parameter")

            bpm_wave = self.bpm_node.render(num_samples_resolved, context, **self.get_params_for_children(params))
            if len(bpm_wave) == 0:
                raise ValueError("Tempo node: BPM evaluation returned no value")
            
            # For dynamic BPM (when bpm_wave varies), use the mean BPM across the chunk
            # This provides better tick accuracy than just using the first sample
            bpm = float(np.mean(bpm_wave))
        
        # Send MIDI clock for internal source
        if self.source == "internal":
            # Enable clock if not already enabled, or update BPM if it changed
            if self.state.last_bpm is None:
                self.midi_output_manager.enable_clock(bpm)
                self.state.last_bpm = bpm
            elif abs(bpm - self.state.last_bpm) > 0.01:  # BPM changed significantly
                self.midi_output_manager.update_bpm(bpm)
                self.state.last_bpm = bpm
            
            # Process this chunk to send MIDI clock messages
            self.midi_output_manager.process_samples(num_samples_resolved)
        
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
        
        # For dynamic BPM, we need to track beats (musical time) rather than wall-clock time
        # We integrate BPM over samples to get accurate beat positions
        start_beats = self.state.beats_since_start
        
        # Add count/accumulator parameters (current bar/beat/eighth/sixteenth/triplet number)
        # These are 1-based counts that increment at each division boundary
        extended_params['bar_number'] = int(start_beats / 4.0) + 1  # Bars in 4/4 = beats / 4
        extended_params['beat_number'] = int(start_beats) + 1  # Beat count
        extended_params['eighth_number'] = int(start_beats / 0.5) + 1  # Eighth note count
        extended_params['sixteenth_number'] = int(start_beats / 0.25) + 1  # Sixteenth note count
        extended_params['triplet_number'] = int(start_beats / (1.0/3.0)) + 1  # Triplet count
        
        # Render BPM for entire chunk to handle dynamic BPM
        bpm_wave = self.bpm_node.render(num_samples_resolved, context, **self.get_params_for_children(params)) if self.bpm_node is not None else np.full(num_samples_resolved, bpm)
        
        # Convert BPM wave to beats-per-sample, then accumulate to get beat position at each sample
        # BPM = beats per minute, so beats per second = BPM / 60, beats per sample = BPM / (60 * SAMPLE_RATE)
        beats_per_sample = bpm_wave / (60.0 * SAMPLE_RATE)
        beat_positions = start_beats + np.cumsum(beats_per_sample)
        
        # Update accumulated beats for next chunk
        end_beats = beat_positions[-1] if len(beat_positions) > 0 else start_beats
        self.state.beats_since_start = end_beats

        def make_ticks(beats_per_period: float) -> np.ndarray:
            """Generate ticks based on beat positions rather than time."""
            ticks = np.zeros(num_samples_resolved, dtype=np.float32)
            if beats_per_period <= 0:
                return ticks

            # Find where beat positions cross integer multiples of the period
            # For example, for beat_tick (beats_per_period=1), we want ticks at beat 0, 1, 2, 3...
            start_period = start_beats / beats_per_period
            end_period = end_beats / beats_per_period
            
            first_n = math.ceil(start_period - 1e-9)
            last_n = math.floor(end_period + 1e-9)
            
            for n in range(first_n, last_n + 1):
                # Find the sample where we cross n * beats_per_period
                target_beat = n * beats_per_period
                # Find first sample where beat_position >= target_beat
                crossing_samples = np.where(beat_positions >= target_beat)[0]
                if len(crossing_samples) > 0:
                    idx = crossing_samples[0]
                    if 0 <= idx < num_samples_resolved:
                        ticks[idx] = 1.0
            return ticks

        # Generate ticks for each division
        # Each type is defined by how many beats per period
        extended_params['bar_tick'] = make_ticks(4.0)      # 4 beats per bar
        extended_params['beat_tick'] = make_ticks(1.0)     # 1 beat per beat
        extended_params['eighth_tick'] = make_ticks(0.5)   # 0.5 beats per eighth note
        extended_params['sixteenth_tick'] = make_ticks(0.25)  # 0.25 beats per sixteenth
        extended_params['triplet_tick'] = make_ticks(1.0/3.0)  # 1/3 beat per triplet
        
        # Multi-bar ticks
        extended_params['two_bar_tick'] = make_ticks(8.0)     # 8 beats per 2 bars
        extended_params['four_bar_tick'] = make_ticks(16.0)   # 16 beats per 4 bars
        extended_params['eight_bar_tick'] = make_ticks(32.0)  # 32 beats per 8 bars
        extended_params['sixteen_bar_tick'] = make_ticks(64.0)  # 64 beats per 16 bars
        
        # Render signal with extended params
        return self.signal_node.render(num_samples, context, num_channels, **extended_params)


TEMPO_DEFINITION = NodeDefinition("tempo", TempoNode, TempoNodeModel)
