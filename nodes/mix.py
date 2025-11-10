"""
The mix node manages multiple audio tracks and combines them.

Features:
- Accepts arbitrary named track arguments (e.g., percussion:, lead:, bass:)
- Each track is rendered and combined into a mixdown
- Always outputs stereo (num_samples, 2) if any child is stereo, otherwise mono
- Responsible for saving individual track stems and final mixdown
- No built-in panning or volume control (use track: node for that)

Example YAML:
  my_song:
    mix:
      percussion:
        osc:
          type: sin
          freq: 100
      lead:
        track:  # Use track node for panning/volume
          signal:
            osc:
              type: sin
              freq: 440
          pan: -0.5
          volume: 0.8
      bass:
        osc:
          type: sin
          freq: 110
"""
from __future__ import annotations
import numpy as np
from typing import Union, Dict
from pydantic import ConfigDict
from types import SimpleNamespace

from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from utils import match_length, empty_stereo, empty_mono, to_mono, is_stereo, to_stereo, add_waves


class MixNodeModel(BaseNodeModel):
    model_config = ConfigDict(extra='allow')  # Accept arbitrary track names


class MixNode(BaseNode):
    def __init__(self, model: MixNodeModel, node_id: str, state=None, do_initialise_state=True):
        super().__init__(model, node_id, state, do_initialise_state)

        # Cache for last rendered track outputs (used for stem export)
        self.last_track_outputs = None

        # Scratch buffers reused during mixing to avoid repeated allocations
        self._mix_buffer_mono: np.ndarray | None = None
        self._mix_buffer_stereo: np.ndarray | None = None
        
        # Parse tracks from __pydantic_extra__
        self.tracks = []  # List of {name, node}
        
        if hasattr(model, '__pydantic_extra__') and model.__pydantic_extra__:
            # Filter out standard BaseNode parameters (duration, bpm, context, etc.)
            standard_params = {'duration', 'bpm', 'context'}
            track_names = [
                name for name in model.__pydantic_extra__.keys() 
                if name not in standard_params
            ]
            
            # Build track configs
            for track_name in track_names:
                track_value = model.__pydantic_extra__[track_name]
                
                # Instantiate track node
                if isinstance(track_value, BaseNodeModel):
                    track_node = self.instantiate_child_node(track_value, track_name)
                else:
                    raise ValueError(f"Track '{track_name}' must be a node, got {type(track_value)}")
                
                self.tracks.append({
                    'name': track_name,
                    'node': track_node,
                })
    
    def get_track_names(self):
        """Return list of track names for file export"""
        return [track['name'] for track in self.tracks]
    
    def _ensure_mix_buffer_mono(self, length: int) -> np.ndarray:
        if self._mix_buffer_mono is None or self._mix_buffer_mono.shape[0] < length:
            self._mix_buffer_mono = np.zeros(length, dtype=np.float32)
        else:
            self._mix_buffer_mono[:length].fill(0)
        return self._mix_buffer_mono[:length]

    def _ensure_mix_buffer_stereo(self, length: int) -> np.ndarray:
        if self._mix_buffer_stereo is None or self._mix_buffer_stereo.shape[0] < length:
            self._mix_buffer_stereo = np.zeros((length, 2), dtype=np.float32)
        else:
            self._mix_buffer_stereo[:length].fill(0)
        return self._mix_buffer_stereo[:length]

    def get_track_outputs(self, num_samples=None, context=None, **params):
        """
        Render all tracks individually and return dict of {track_name: array}.
        Used for exporting individual track stems.
        
        Renders child nodes and returns them as-is (mono or stereo).
        """
        track_outputs = {}

        for track in self.tracks:
            # Render child node (may return mono or stereo)
            signal = track['node'].render(num_samples, context, **params)
            track_outputs[track['name']] = signal
        
        return track_outputs
    
    def _do_render(self, num_samples=None, context=None, **params):
        """
        Render all tracks and mix them together.
        Also caches track outputs in self.last_track_outputs for stem export.
        
        Output format depends on children:
        - If all children are mono: returns mono (1D array)
        - If any child is stereo: returns stereo (2D array)
        
        Returns:
            1D array (mono) or 2D array of shape (num_samples, 2) (stereo)
        """
        track_outputs = self.get_track_outputs(num_samples, context, **params)
        
        # Cache the track outputs for stem export
        self.last_track_outputs = track_outputs
        
        # Check if all tracks are empty (finished rendering)
        if all(len(output) == 0 for output in track_outputs.values()):
            # Check if we should return stereo or mono empty based on previous renders
            # For simplicity, return mono empty (parent will handle conversion if needed)
            return empty_mono()
        
        # Find the maximum length among non-empty tracks
        non_empty_lengths = [len(output) for output in track_outputs.values() if len(output) > 0]
        if not non_empty_lengths:
            return empty_mono()
        
        max_length = max(non_empty_lengths)
        
        # Determine if we need stereo output (if any track is stereo)
        has_stereo = any(is_stereo(output) for output in track_outputs.values() if len(output) > 0)
        
        # Mix all tracks together
        if has_stereo:
            # Stereo mixing
            mixed = self._ensure_mix_buffer_stereo(max_length)
            
            for track in self.tracks:
                track_name = track['name']
                signal = track_outputs.get(track_name)

                if signal is None or len(signal) == 0:
                    continue

                frames = min(len(signal), max_length)
                
                # Convert to stereo if needed
                if not is_stereo(signal):
                    signal = to_stereo(signal)
                
                mixed[:frames] += signal[:frames]
        else:
            # Mono mixing
            mixed = self._ensure_mix_buffer_mono(max_length)
            
            for track in self.tracks:
                track_name = track['name']
                signal = track_outputs.get(track_name)

                if signal is None or len(signal) == 0:
                    continue

                frames = min(len(signal), max_length)
                mixed[:frames] += signal[:frames]

        return mixed.copy()


MIX_DEFINITION = NodeDefinition(
    name="mix",
    model=MixNodeModel,
    node=MixNode
)
