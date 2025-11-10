"""
The tracks node is a special root-level node that manages multiple audio tracks with stereo panning.

Features:
- Accepts arbitrary named track arguments (e.g., percussion:, lead:, bass:)
- Each track can have an associated _pan parameter (e.g., lead_pan: -0.5)
- Pan values range from -1 (full left) to 1 (full right), default is 0 (center)
- Pan can be a scalar or WavableValue for dynamic panning
- Always outputs stereo (num_samples, 2) regardless of panning
- Responsible for saving individual track stems and final mixdown
- Uses equal-power panning for smooth stereo imaging

Example YAML:
  my_song:
    tracks:
      percussion:
        osc:
          type: sin
          freq: 100
      lead:
        osc:
          type: sin
          freq: 440
      lead_pan: -0.5  # Pan lead 50% to the left
      bass:
        osc:
          type: sin
          freq: 110
      bass_pan:  # Dynamic panning with LFO
        osc:
          type: sin
          freq: 0.5
          range: [-1, 1]
"""
from __future__ import annotations
import numpy as np
from typing import Union, Dict
from pydantic import ConfigDict
from types import SimpleNamespace

from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.node_utils.panning import apply_panning
from utils import match_length, empty_stereo, to_mono, is_stereo


class TracksNodeModel(BaseNodeModel):
    model_config = ConfigDict(extra='allow')  # Accept arbitrary track names and _pan parameters


class TracksNode(BaseNode):
    def __init__(self, model: TracksNodeModel, node_id: str, state=None, do_initialise_state=True):
        super().__init__(model, node_id, state, do_initialise_state)

        # Cache for last rendered track outputs (used for stem export)
        self.last_track_outputs = None

        # Scratch buffers reused during mixing to avoid repeated allocations
        self._mix_buffer: np.ndarray | None = None
        self._stereo_scratch: np.ndarray | None = None
        self._pan_cache: dict[int, tuple[int, np.ndarray]] = {}
        
        # Parse tracks and their associated pan values from __pydantic_extra__
        self.tracks = []  # List of {name, node, pan_node_or_value}
        
        if hasattr(model, '__pydantic_extra__') and model.__pydantic_extra__:
            # First pass: identify all track names (non-_pan/_vol parameters and non-base-model-fields)
            # Filter out: _pan and _vol suffixes and standard BaseNode parameters (duration, amp, etc.)
            standard_params = {'duration', 'bpm', 'context'}
            track_names = [
                name for name in model.__pydantic_extra__.keys() 
                if not name.endswith('_pan') and not name.endswith('_vol') and name not in standard_params
            ]
            
            # Second pass: build track configs with their pan and vol values
            for track_name in track_names:
                track_value = model.__pydantic_extra__[track_name]
                pan_param_name = f"{track_name}_pan"
                pan_value = model.__pydantic_extra__.get(pan_param_name, 0)  # Default to center
                vol_param_name = f"{track_name}_vol"
                vol_value = model.__pydantic_extra__.get(vol_param_name, 1.0)  # Default to unity gain
                
                # Instantiate track node
                if isinstance(track_value, BaseNodeModel):
                    track_node = self.instantiate_child_node(track_value, track_name)
                else:
                    raise ValueError(f"Track '{track_name}' must be a node, got {type(track_value)}")
                
                # Handle pan value (can be scalar or node for dynamic panning)
                if isinstance(pan_value, BaseNodeModel):
                    pan_node = self.instantiate_child_node(pan_value, pan_param_name)
                    is_pan_dynamic = True
                else:
                    pan_node = float(pan_value)
                    is_pan_dynamic = False
                
                # Handle vol value (can be scalar or node for dynamic volume)
                if isinstance(vol_value, BaseNodeModel):
                    vol_node = self.instantiate_child_node(vol_value, vol_param_name)
                    is_vol_dynamic = True
                else:
                    vol_node = float(vol_value)
                    is_vol_dynamic = False
                
                self.tracks.append({
                    'name': track_name,
                    'node': track_node,
                    'pan': pan_node,
                    'is_pan_dynamic': is_pan_dynamic,
                    'vol': vol_node,
                    'is_vol_dynamic': is_vol_dynamic
                })
    
    def get_track_names(self):
        """Return list of track names for file export"""
        return [track['name'] for track in self.tracks]
    
    def _ensure_mix_buffer(self, length: int) -> np.ndarray:
        if self._mix_buffer is None or self._mix_buffer.shape[0] < length:
            self._mix_buffer = np.zeros((length, 2), dtype=np.float32)
        else:
            self._mix_buffer[:length].fill(0)
        return self._mix_buffer[:length]

    def _ensure_stereo_scratch(self, length: int) -> np.ndarray:
        if self._stereo_scratch is None or self._stereo_scratch.shape[0] < length:
            self._stereo_scratch = np.zeros((length, 2), dtype=np.float32)
        return self._stereo_scratch[:length]

    def _get_pan_envelope(self, pan_entry, length: int, context, params, pan_cache):
        cache_key = id(pan_entry)
        cached = pan_cache.get(cache_key)
        if cached and cached[0] >= length:
            return cached[1][:length]

        pan_value = pan_entry.render(length, context, **params)
        pan_value = match_length(pan_value, length)
        pan_cache[cache_key] = (len(pan_value), pan_value)
        return pan_value

    def get_track_outputs(self, num_samples=None, context=None, **params):
        """
        Render all tracks individually and return dict of {track_name: stereo_array}.
        Used for exporting individual track stems.
        
        Renders child nodes and handles mono/stereo conversion. If a child returns:
        - Mono (1D array): Apply panning to create stereo
        - Stereo (2D array): Convert to mono first, then apply panning (unless pan is center)
        """
        track_outputs = {}
        self._pan_cache.clear()

        for track in self.tracks:
            # Render child node (may return mono or stereo)
            signal = track['node'].render(num_samples, context, **params)
            
            # If signal is empty, this track is done
            if len(signal) == 0:
                track_outputs[track['name']] = empty_stereo()
                continue
            
            # Check if we got mono or stereo
            if is_stereo(signal):
                # Signal is already stereo
                # If we have a pan setting (not default center), mix to mono first then apply panning
                # This ensures _pan always works, even for stereo sources
                has_pan = track['is_pan_dynamic'] or track['pan'] != 0
                
                if has_pan:
                    # Mix stereo to mono (average of left and right channels)
                    mono_signal = to_mono(signal)
                    
                    # Get pan value (static or dynamic)
                    if track['is_pan_dynamic']:
                        pan_value = self._get_pan_envelope(track['pan'], len(mono_signal), context, params, self._pan_cache)
                    else:
                        pan_value = track['pan']

                    # Apply panning to create new stereo signal
                    stereo_signal = apply_panning(mono_signal, pan_value)
                else:
                    # No pan setting - use stereo signal as-is
                    stereo_signal = signal
            else:
                # Mono - apply panning to create stereo
                # Get pan value (static or dynamic)
                if track['is_pan_dynamic']:
                    pan_value = self._get_pan_envelope(track['pan'], len(signal), context, params, self._pan_cache)
                else:
                    pan_value = track['pan']

                # Apply panning to create stereo
                stereo_signal = apply_panning(signal, pan_value)
            
            track_outputs[track['name']] = stereo_signal
        
        return track_outputs
    
    def _do_render(self, num_samples=None, context=None, **params):
        """
        Render all tracks and mix them to stereo output.
        Also caches track outputs in self.last_track_outputs for stem export.
        Volume (_vol) is applied only to the mixdown, not to individual track exports.
        
        Note: TracksNode always outputs stereo.
        
        Returns:
            2D array of shape (num_samples, 2) with mixed stereo audio
        """
        track_outputs = self.get_track_outputs(num_samples, context, **params)
        
        # Cache the track outputs for stem export (without volume applied)
        self.last_track_outputs = track_outputs
        
        # Check if all tracks are empty (finished rendering)
        if all(len(output) == 0 for output in track_outputs.values()):
            return empty_stereo()
        
        # Find the maximum length among non-empty tracks (we want to render until ALL tracks finish)
        non_empty_lengths = [len(output) for output in track_outputs.values() if len(output) > 0]
        if not non_empty_lengths:
            return empty_stereo()
        
        max_length = max(non_empty_lengths)
        
        # Mix all tracks together with volume applied
        # Start with zeros
        mixed = self._ensure_mix_buffer(max_length)

        for track in self.tracks:
            track_name = track['name']
            stereo_signal = track_outputs.get(track_name)

            if stereo_signal is None or len(stereo_signal) == 0:
                continue

            frames = min(len(stereo_signal), max_length)
            contribution = stereo_signal[:frames]

            # Get volume value (static or dynamic)
            if track['is_vol_dynamic']:
                vol_value = track['vol'].render(frames, context, **params)
                vol_value = match_length(vol_value, frames)
                scratch = self._ensure_stereo_scratch(frames)
                np.multiply(contribution, vol_value[:, np.newaxis], out=scratch)
                contribution = scratch
            else:
                vol = track['vol']
                if vol != 1.0:
                    scratch = self._ensure_stereo_scratch(frames)
                    np.multiply(contribution, vol, out=scratch)
                    contribution = scratch

            mixed[:frames] += contribution

        return mixed


TRACKS_DEFINITION = NodeDefinition(
    name="tracks",
    model=TracksNodeModel,
    node=TracksNode
)
