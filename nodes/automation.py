"""
Automation Node - Sequences control values with different interpolation modes.

Unlike the sequencer node which is designed for playing sounds, the automation node
is optimized for sequencing control values (like frequency, amplitude, filter cutoff, etc.)
though both work with waves since everything is a wave in this system.

Features:
- Steps can be scalars, expressions, nodes, or interpolated lists (any WavableValue)
- Three interpolation modes: step, ramp, and overlap
- Interval parameter controls step duration
- None steps are allowed and will be skipped (previous value holds)
- Supports repeating the automation sequence

Modes:
1. STEP mode: Holds constant values between defined steps
   Example: step 2 = 5, step 5 = 10 → outputs 5 from step 2-5, then 10 from step 5+

2. RAMP mode: Linear interpolation between defined steps  
   Example: step 2 = 5, step 5 = 10 → gradually interpolates from 5 to 10 over steps 2-5

3. OVERLAP mode: Crossfade between values with configurable overlap time
   The previous value continues rendering while the new value fades in over the specified
   overlap duration. The crossfade begins at the start of each new step with a value.

Example usage:
```yaml
my_sound:
  osc:
    type: sin
    freq:
      automation:
        interval: 0.5        # Each step is 0.5 seconds
        mode: step           # or 'ramp' or 'overlap'
        overlap: 0.1         # Crossfade time (only used in overlap mode)
        repeat: 2            # Repeat the sequence twice
        steps:
          - 220              # Step 0: A3
          -                  # Step 1: (None, holds 220)
          - 440              # Step 2: A4
          -                  # Step 3: (None, holds 440)
          - 880              # Step 4: A5
```

You can also use nodes as step values:
```yaml
freq:
  automation:
    interval: 1
    mode: step
    steps:
      - osc:                # Step 0: LFO from 200-400 Hz
          type: sin
          freq: 2
          range: [200, 400]
      -                     # Step 1: (None, continues LFO)
      - osc:                # Step 2: LFO from 400-800 Hz
          type: sin
          freq: 5
          range: [400, 800]
```
"""

from __future__ import annotations
from enum import Enum
from typing import List, Optional
import numpy as np
from pydantic import ConfigDict
from config import SAMPLE_RATE
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.wavable_value import WavableValue


class AutomationMode(str, Enum):
    STEP = "step"
    RAMP = "ramp"
    OVERLAP = "overlap"


class AutomationModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    steps: List[Optional[WavableValue]]
    interval: WavableValue = 1.0
    mode: AutomationMode = AutomationMode.STEP
    overlap: float = 0.1  # Crossfade time in seconds for overlap mode
    repeat: int = 1


class AutomationNode(BaseNode):
    def __init__(self, model: AutomationModel, node_id: str, state, hot_reload=False):
        super().__init__(model, node_id, state, hot_reload)
        self.model = model
        self.mode = model.mode
        self.overlap_time = model.overlap
        self.repeat = model.repeat
        
        # Create interval node
        self.interval_node = self.instantiate_child_node(model.interval, "interval")

        
        # Persistent state for realtime playback (survives hot reload)
        if not hot_reload:
            self.state.current_repeat = 0
            self.state.current_step = 0
            self.state.time_in_current_step = 0  # Time elapsed in current step (seconds)
            self.state.sequence_complete = False
            # For overlap mode, track the previous step's output for crossfading
            self.state.prev_step_buffer = None  # Buffer to store previous step's output during crossfade
            self.state.prev_step_index = None
            self.state.crossfade_progress = 0  # Samples rendered in current crossfade

        self.step_nodes = []
        for step_index, step_value in enumerate(model.steps):
            if step_value is not None:
                step_node = self.instantiate_child_node(step_value, f"step_{step_index}", self.state.current_repeat)
                self.step_nodes.append(step_node)
            else:
                self.step_nodes.append(None)

    def _find_next_non_none_step(self, start_index):
        """Find the next step index that has a value (not None)."""
        for i in range(start_index, len(self.step_nodes)):
            if self.step_nodes[i] is not None:
                return i
        return None
    
    def _find_prev_non_none_step(self, start_index):
        """Find the previous step index that has a value (not None)."""
        for i in range(start_index - 1, -1, -1):
            if self.step_nodes[i] is not None:
                return i
        return None

    def _calculate_total_samples(self, interval):
        """Calculate total number of samples for the entire automation sequence"""
        total_steps = len(self.step_nodes)
        total_duration = total_steps * interval * self.repeat
        return int(total_duration * SAMPLE_RATE)

    def _do_render(self, num_samples=None, context=None, **params):
        # Evaluate interval for this chunk
        num_samples_resolved = self.resolve_num_samples(num_samples)
        if num_samples_resolved is None:
            num_samples_resolved = num_samples
        
        if num_samples_resolved is None:
            # Non-realtime mode - calculate total duration
            interval_wave = self.interval_node.render(1, context, **self.get_params_for_children(params))
            interval = float(interval_wave[0]) if len(interval_wave) > 0 else 1.0
            num_samples_resolved = self._calculate_total_samples(interval)
            if num_samples_resolved == 0:
                return np.array([])
            self._last_chunk_samples = num_samples_resolved
        
        # Get interval value
        interval_wave = self.interval_node.render(num_samples_resolved, context, **self.get_params_for_children(params))
        interval = float(interval_wave[0]) if len(interval_wave) > 0 else 1.0
        
        if len(self.step_nodes) == 0:
            return np.zeros(num_samples_resolved, dtype=np.float32)
        
        output_wave = np.zeros(num_samples_resolved, dtype=np.float32)
        samples_written = 0
        samples_per_step = int(interval * SAMPLE_RATE)
        
        while samples_written < num_samples_resolved:
            # Check if we've completed the current repeat
            if self.state.current_step >= len(self.step_nodes):
                self.state.current_repeat += 1
                if self.state.current_repeat >= self.repeat:
                    # We've completed all repeats - stay at the last step forever
                    self.state.sequence_complete = True
                    self.state.current_step = len(self.step_nodes) - 1
                    self.state.time_in_current_step = 0
                else:
                    # Start next repeat from the beginning
                    self.state.current_step = 0
                    self.state.time_in_current_step = 0
                    self.state.prev_step_buffer = None
                    self.state.prev_step_index = None
                    self.state.crossfade_progress = 0
                    continue
            
            # Calculate how many samples we can render from current step
            time_remaining_in_step = interval - self.state.time_in_current_step
            samples_remaining_in_step = int(time_remaining_in_step * SAMPLE_RATE)
            samples_to_render = min(samples_remaining_in_step, num_samples_resolved - samples_written)
            
            if samples_to_render <= 0:
                # Move to next step
                self.state.current_step += 1
                self.state.time_in_current_step = 0
                self.state.prev_step_buffer = None
                self.state.crossfade_progress = 0
                continue
            
            # Render based on mode
            if self.mode == AutomationMode.STEP:
                chunk = self._render_step_mode(samples_to_render, context, params)
            elif self.mode == AutomationMode.RAMP:
                chunk = self._render_ramp_mode(samples_to_render, interval, context, params)
            elif self.mode == AutomationMode.OVERLAP:
                chunk = self._render_overlap_mode(samples_to_render, interval, context, params)
            else:
                chunk = np.zeros(samples_to_render, dtype=np.float32)
            
            # Add to output
            output_wave[samples_written:samples_written + len(chunk)] = chunk
            samples_written += len(chunk)
            
            # Update time in current step
            self.state.time_in_current_step += len(chunk) / SAMPLE_RATE
            
            # Check if we should move to next step
            if self.state.time_in_current_step >= interval:
                self.state.current_step += 1
                self.state.time_in_current_step = 0
                self.state.prev_step_buffer = None
                self.state.crossfade_progress = 0
        
        return output_wave

    def _render_step_mode(self, num_samples, context, params):
        """Step mode: hold constant value until next step with a value."""
        # Find the current active step (most recent non-None step at or before current position)
        active_step_index = self._find_prev_non_none_step(self.state.current_step + 1)
        if active_step_index is None:
            active_step_index = self._find_next_non_none_step(self.state.current_step)
        
        if active_step_index is None or self.step_nodes[active_step_index] is None:
            return np.zeros(num_samples, dtype=np.float32)
        
        # Render from the active step node
        result = self.step_nodes[active_step_index].render(num_samples, context, **self.get_params_for_children(params))
        
        # Handle empty arrays (node finished rendering)
        if len(result) == 0:
            return np.zeros(num_samples, dtype=np.float32)
        
        # If node returned fewer samples than requested, pad with last value
        if len(result) < num_samples:
            last_value = result[-1] if len(result) > 0 else 0
            padding = np.full(num_samples - len(result), last_value)
            result = np.concatenate([result, padding])
        
        return result

    def _render_ramp_mode(self, num_samples, interval, context, params):
        """Ramp mode: linear interpolation between steps with values."""
        # Find current and next non-None steps
        current_value_step = self._find_prev_non_none_step(self.state.current_step + 1)
        if current_value_step is None:
            current_value_step = self._find_next_non_none_step(self.state.current_step)
        
        next_value_step = self._find_next_non_none_step(self.state.current_step + 1)
        
        # If no next value in current sequence and we have more repeats, look at the start
        if next_value_step is None and self.state.current_repeat + 1 < self.repeat:
            next_value_step = self._find_next_non_none_step(0)
        
        # If no values found, return zeros
        if current_value_step is None:
            return np.zeros(num_samples, dtype=np.float32)
        
        # If no next value, just hold current value
        if next_value_step is None:
            result = self.step_nodes[current_value_step].render(num_samples, context, **self.get_params_for_children(params))
            if len(result) == 0:
                return np.zeros(num_samples, dtype=np.float32)
            if len(result) < num_samples:
                last_value = result[-1] if len(result) > 0 else 0
                padding = np.full(num_samples - len(result), last_value)
                result = np.concatenate([result, padding])
            return result
        
        # Calculate interpolation factor
        # If next_value_step is before current_value_step, we're wrapping to next repeat
        if next_value_step <= current_value_step:
            # Wrapping to start of sequence
            steps_between = (len(self.step_nodes) - current_value_step) + next_value_step
        else:
            steps_between = next_value_step - current_value_step
        current_position_in_ramp = self.state.current_step - current_value_step + (self.state.time_in_current_step / interval)
        
        # Render both values
        current_value = self.step_nodes[current_value_step].render(num_samples, context, **self.get_params_for_children(params))
        next_value = self.step_nodes[next_value_step].render(num_samples, context, **self.get_params_for_children(params))
        
        # Handle empty arrays
        if len(current_value) == 0 and len(next_value) == 0:
            return np.zeros(num_samples, dtype=np.float32)
        elif len(current_value) == 0:
            current_value = np.zeros(num_samples, dtype=np.float32)
        elif len(next_value) == 0:
            next_value = np.zeros(num_samples, dtype=np.float32)
        
        # Pad if necessary
        if len(current_value) < num_samples:
            last_value = current_value[-1] if len(current_value) > 0 else 0
            padding = np.full(num_samples - len(current_value), last_value)
            current_value = np.concatenate([current_value, padding])
        
        if len(next_value) < num_samples:
            last_value = next_value[-1] if len(next_value) > 0 else 0
            padding = np.full(num_samples - len(next_value), last_value)
            next_value = np.concatenate([next_value, padding])
        
        # Create interpolation weights for each sample in the chunk
        sample_positions = np.arange(num_samples) / SAMPLE_RATE
        interpolation_factors = (current_position_in_ramp + sample_positions / interval) / steps_between
        interpolation_factors = np.clip(interpolation_factors, 0, 1)
        
        # Linear interpolation
        result = current_value * (1 - interpolation_factors) + next_value * interpolation_factors
        return result

    def _render_overlap_mode(self, num_samples, interval, context, params):
        """Overlap mode: crossfade between steps with configurable overlap time.
        
        When a new step with a value is reached, the previous value continues rendering
        while the new value fades in over the overlap duration. The crossfade happens
        at the beginning of the new step.
        """
        overlap_samples = int(self.overlap_time * SAMPLE_RATE)
        
        # Find current active step (most recent non-None step at or before current position)
        current_value_step = self._find_prev_non_none_step(self.state.current_step + 1)
        if current_value_step is None:
            current_value_step = self._find_next_non_none_step(self.state.current_step)
        
        if current_value_step is None:
            return np.zeros(num_samples, dtype=np.float32)
        
        # Check if we're at the exact step that has a value (start of a new value step)
        at_value_step = (self.state.current_step < len(self.step_nodes) and 
                        self.step_nodes[self.state.current_step] is not None)
        
        # If we just arrived at a new step with a value, initiate crossfade
        if at_value_step and self.state.time_in_current_step == 0:
            # Find the previous value step for crossfading
            prev_step = self._find_prev_non_none_step(self.state.current_step)
            
            # If we're at step 0 and wrapping from a previous repeat, look at the end
            if prev_step is None and self.state.current_step == 0 and self.state.current_repeat > 0:
                prev_step = self._find_prev_non_none_step(len(self.step_nodes))
            
            if prev_step is not None and prev_step != self.state.current_step:
                self.state.prev_step_index = prev_step
                self.state.crossfade_progress = 0
        
        # Calculate current position within the step (in samples)
        samples_into_step = int(self.state.time_in_current_step * SAMPLE_RATE)
        
        # Determine if we're in a crossfade period
        in_crossfade = (self.state.prev_step_index is not None and 
                       samples_into_step < overlap_samples and
                       self.state.prev_step_index != current_value_step)
        
        # Render current step
        current_output = self.step_nodes[current_value_step].render(num_samples, context, **self.get_params_for_children(params))
        
        # Handle empty array from current step
        if len(current_output) == 0:
            current_output = np.zeros(num_samples, dtype=np.float32)
        elif len(current_output) < num_samples:
            last_value = current_output[-1] if len(current_output) > 0 else 0
            padding = np.full(num_samples - len(current_output), last_value)
            current_output = np.concatenate([current_output, padding])
        
        if in_crossfade and self.step_nodes[self.state.prev_step_index] is not None:
            # Render previous step
            prev_output = self.step_nodes[self.state.prev_step_index].render(num_samples, context, **self.get_params_for_children(params))
            
            # Handle empty array from previous step
            if len(prev_output) == 0:
                prev_output = np.zeros(num_samples, dtype=np.float32)
            elif len(prev_output) < num_samples:
                last_value = prev_output[-1] if len(prev_output) > 0 else 0
                padding = np.full(num_samples - len(prev_output), last_value)
                prev_output = np.concatenate([prev_output, padding])
            
            # Calculate how much of this chunk is in the crossfade
            crossfade_remaining = overlap_samples - samples_into_step
            samples_to_crossfade = min(num_samples, crossfade_remaining)
            
            # Create fade curves - only apply crossfade to the relevant portion
            result = current_output.copy()
            
            if samples_to_crossfade > 0:
                # Calculate fade factors for each sample in the crossfade region
                fade_positions = np.arange(samples_to_crossfade) + samples_into_step
                fade_factors = fade_positions / overlap_samples
                fade_out = 1 - fade_factors
                fade_in = fade_factors
                
                # Apply crossfade
                result[:samples_to_crossfade] = (prev_output[:samples_to_crossfade] * fade_out + 
                                                 current_output[:samples_to_crossfade] * fade_in)
            
            # After crossfade ends, clear prev_step_index
            if samples_into_step + num_samples >= overlap_samples:
                # We've passed the end of the crossfade
                self.state.prev_step_index = None
            
            return result
        else:
            # No crossfade, just return current output
            return current_output


AUTOMATION_DEFINITION = NodeDefinition(
    name='automation',
    model=AutomationModel,
    node=AutomationNode
)
