from __future__ import annotations
import numpy as np
from typing import List, Optional, Union
from pydantic import ConfigDict, model_validator
from config import SAMPLE_RATE
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.wavable_value import WavableValue


# Snap node: Snaps signal values to the closest value in a provided list.
# Useful for quantizing continuous control signals (like MIDI CC) to discrete values (like notes).
# The glide parameter adds portamento-style transitions between snapped values.
#
# Values can be specified in two ways:
# 1. Explicit list: values: [100, 200, 400, 800] (can be WavableValues)
# 2. Range + interval: range: [min, max], interval: step (all can be WavableValues)
#
# All parameters support WavableValues for dynamic modulation.
class SnapNodeModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    signal: BaseNodeModel  # The input signal to snap
    values: Optional[List[WavableValue]] = None  # Explicit list of values to snap to
    range: Optional[List[WavableValue]] = None  # [min, max] range to generate values from
    interval: Optional[WavableValue] = None  # Step size for generating values from range
    glide: WavableValue = 0  # Glide time in seconds (0 = instant snap)
    
    @model_validator(mode='after')
    def validate_values_or_range(self):
        """Ensure either values or (range + interval) is provided."""
        if self.values is None and (self.range is None or self.interval is None):
            raise ValueError("Must provide either 'values' or both 'range' and 'interval'")
        if self.values is not None and (self.range is not None or self.interval is not None):
            raise ValueError("Cannot provide both 'values' and 'range'/'interval'")
        if self.range is not None and len(self.range) != 2:
            raise ValueError("'range' must be a list of exactly 2 values [min, max]")
        return self


class SnapNode(BaseNode):
    def __init__(self, model: SnapNodeModel, node_id: str, state=None, hot_reload=False):
        from nodes.node_utils.instantiate_node import instantiate_node
        from nodes.wavable_value import WavableValueNode, WavableValueModel
        super().__init__(model, node_id, state, hot_reload)
        self.signal_node = self.instantiate_child_node(model.signal, "signal")
        
        # Determine if we're using explicit values or range + interval
        self.use_explicit_values = model.values is not None
        
        if self.use_explicit_values:
            # Setup value nodes
            self.value_nodes = []
            self.static_values = []
            self.are_all_values_static = True
            
            for val_idx, value in enumerate(model.values):
                if isinstance(value, (int, float)):
                    # Static scalar
                    self.static_values.append(float(value))
                    self.value_nodes.append(None)
                else:
                    # Dynamic value
                    self.are_all_values_static = False
                    self.value_nodes.append(self.instantiate_child_node(value, f"values_{val_idx}"))
            
            # If all values are static, pre-compute snap values
            if self.are_all_values_static:
                self.snap_values = np.array(sorted(self.static_values), dtype=np.float32)
        else:
            self.range_min_node = self.instantiate_child_node(model.range[0], "range_min")
            self.range_max_node = self.instantiate_child_node(model.range[1], "range_max")
            self.interval_node = self.instantiate_child_node(model.interval, "interval")
            
            # Check if range and interval are static
            self.range_interval_static = (
                self._is_static_value(model.range[0]) and
                self._is_static_value(model.range[1]) and
                self._is_static_value(model.interval)
            )
            
            # Pre-compute if static
            if self.range_interval_static:
                min_val = float(model.range[0]) if isinstance(model.range[0], (int, float)) else float(model.range[0])
                max_val = float(model.range[1]) if isinstance(model.range[1], (int, float)) else float(model.range[1])
                interval_val = float(model.interval) if isinstance(model.interval, (int, float)) else float(model.interval)
                snap_values = np.arange(min_val, max_val + interval_val / 2, interval_val)
                self.snap_values = np.array(sorted(snap_values), dtype=np.float32)
        
        # Setup glide node
        self.glide_node = self.instantiate_child_node(model.glide, "glide")
        self.glide_is_static = self._is_static_value(model.glide)
        if self.glide_is_static:
            self.static_glide = float(model.glide) if isinstance(model.glide, (int, float)) else 0
        
        self.last_output = None  # Track last output for glide transitions
    
    def _is_static_value(self, value: WavableValue) -> bool:
        """Check if a value is static (scalar) or dynamic (node/expression/list)."""
        return isinstance(value, (int, float))
    
    def _do_render(self, num_samples=None, context=None, **params):
        num_samples = self.resolve_num_samples(num_samples)
        
        # Render the input signal
        if num_samples is None:
            signal_wave = self.render_full_child_signal(
                self.signal_node, context, **self.get_params_for_children(params)
            )
            if len(signal_wave) == 0:
                return np.array([], dtype=np.float32)
        else:
            signal_wave = self.signal_node.render(
                num_samples, context, **self.get_params_for_children(params)
            )
            if len(signal_wave) == 0:
                return np.array([], dtype=np.float32)
        
        actual_num_samples = len(signal_wave)
        child_params = self.get_params_for_children(params)
        
        # Get snap values for this chunk (static or dynamic)
        if self.use_explicit_values:
            if self.are_all_values_static:
                # Use pre-computed static values (fast path)
                snap_values = self.snap_values
            else:
                # Render dynamic values (slower path)
                rendered_values = []
                for i, node in enumerate(self.value_nodes):
                    if node is None:
                        # Static value
                        rendered_values.append(self.static_values[i])
                    else:
                        # Dynamic value - render and take first sample
                        val_wave = node.render(actual_num_samples, context, **child_params)
                        rendered_values.append(val_wave[0] if len(val_wave) > 0 else 0)
                snap_values = np.array(sorted(rendered_values), dtype=np.float32)
        else:
            if self.range_interval_static:
                # Use pre-computed static values (fast path)
                snap_values = self.snap_values
            else:
                # Render dynamic range and interval (slower path)
                min_wave = self.range_min_node.render(actual_num_samples, context, **child_params)
                max_wave = self.range_max_node.render(actual_num_samples, context, **child_params)
                interval_wave = self.interval_node.render(actual_num_samples, context, **child_params)
                
                # Take first sample of each for this chunk
                min_val = min_wave[0] if len(min_wave) > 0 else 0
                max_val = max_wave[0] if len(max_wave) > 0 else 1
                interval_val = interval_wave[0] if len(interval_wave) > 0 else 0.1
                
                # Generate snap values
                snap_values_list = np.arange(min_val, max_val + interval_val / 2, interval_val)
                snap_values = np.array(sorted(snap_values_list), dtype=np.float32)
        
        # Snap each value to the closest value in snap_values
        snapped = self._snap_to_values(signal_wave, snap_values)
        
        # Get glide value (static or dynamic)
        if self.glide_is_static:
            glide_time = self.static_glide
        else:
            glide_wave = self.glide_node.render(actual_num_samples, context, **child_params)
            glide_time = glide_wave[0] if len(glide_wave) > 0 else 0
        
        # Apply glide if requested
        if glide_time > 0:
            snapped = self._apply_glide(snapped, snap_values, glide_time)

        return snapped
    
    def _snap_to_values(self, signal: np.ndarray, snap_values: np.ndarray) -> np.ndarray:
        """Snap each value in the signal to the closest value in snap_values."""
        # Vectorized snapping using broadcasting
        # For each signal value, compute distance to all snap values
        # Shape: (len(signal), len(snap_values))
        distances = np.abs(signal[:, np.newaxis] - snap_values[np.newaxis, :])
        
        # Find index of closest snap value for each signal value
        closest_indices = np.argmin(distances, axis=1)
        
        # Get the snapped values
        snapped = snap_values[closest_indices]
        
        return snapped.astype(np.float32)
    
    def _apply_glide(self, signal: np.ndarray, snap_values: np.ndarray, glide_time: float) -> np.ndarray:
        """Apply glide/portamento transitions between snapped values.
        
        Glide is specified in seconds - the time it takes to traverse the maximum
        possible range between snap values.
        
        Optimized implementation using change-point detection and vectorized interpolation.
        This is much faster than sample-by-sample processing, especially when the signal
        has long constant segments (typical for MIDI CC input).
        """
        # Calculate max range between snap values
        max_range = float(np.max(snap_values) - np.min(snap_values))
        
        # Convert glide time (seconds) to change per sample
        if glide_time <= 0 or max_range <= 0:
            return signal
        
        glide_per_sample = max_range / (glide_time * SAMPLE_RATE)
        
        # Initialize with last output from previous chunk, or first value
        if self.last_output is not None:
            current_value = self.last_output
        else:
            current_value = signal[0]
        
        # Find indices where the signal changes value
        # This allows us to process constant segments efficiently
        changes = np.concatenate([[0], np.where(np.diff(signal) != 0)[0] + 1, [len(signal)]])
        
        glided = np.empty_like(signal, dtype=np.float32)
        
        # Process each constant segment
        for i in range(len(changes) - 1):
            start_idx = changes[i]
            end_idx = changes[i + 1]
            target_value = signal[start_idx]
            
            # Calculate how many samples needed to reach target from current value
            distance = abs(target_value - current_value)
            samples_needed = int(np.ceil(distance / glide_per_sample))
            
            if samples_needed == 0:
                # Already at target - fill segment with target value
                glided[start_idx:end_idx] = target_value
                current_value = target_value
            else:
                segment_len = end_idx - start_idx
                
                if samples_needed < segment_len:
                    # Will reach target before segment ends
                    # Create linear ramp to target, then hold
                    ramp = np.linspace(current_value, target_value, samples_needed + 1, dtype=np.float32)[1:]
                    glided[start_idx:start_idx + samples_needed] = ramp
                    glided[start_idx + samples_needed:end_idx] = target_value
                    current_value = target_value
                else:
                    # Still gliding at end of segment
                    # Create ramp that moves toward target by glide_per_sample each sample
                    direction = np.sign(target_value - current_value)
                    ramp = current_value + direction * glide_per_sample * np.arange(1, segment_len + 1, dtype=np.float32)
                    glided[start_idx:end_idx] = ramp
                    current_value = ramp[-1]
        
        # Store last output for next chunk
        self.last_output = current_value
        
        return glided


SNAP_DEFINITION = NodeDefinition("snap", SnapNode, SnapNodeModel)
