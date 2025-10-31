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
import math
from typing import List, Optional
import numpy as np
from pydantic import ConfigDict
from config import SAMPLE_RATE
from utils import time_to_samples, samples_to_time, get_last_or_default, detect_triggers
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
    repeat: int = math.inf
    swing: WavableValue = 0
    trigger: Optional[WavableValue] = None  # Optional trigger signal to advance steps
    reset: Optional[WavableValue] = None  # Optional reset signal to reset to step 0


class AutomationNode(BaseNode):
    def __init__(self, model: AutomationModel, node_id: str, state, do_initialise_state=True):
        super().__init__(model, node_id, state, do_initialise_state)
        self.model = model
        self.mode = model.mode
        self.overlap_time = model.overlap
        self.repeat = model.repeat
        
        # Create interval and swing nodes
        self.interval_node = self.instantiate_child_node(model.interval, "interval")
        self.swing_node = self.instantiate_child_node(model.swing, "swing")
        
        # Optional trigger and reset nodes
        self.trigger_node = self.instantiate_child_node(model.trigger, "trigger") if model.trigger is not None else None
        self.reset_node = self.instantiate_child_node(model.reset, "reset") if model.reset is not None else None

        
        # Persistent state for realtime playback (survives hot reload)
        if do_initialise_state:
            self.state.current_repeat = 0
            self.state.current_step = 0
            self.state.next_step_to_trigger = 0  # In trigger mode, the next step that will be triggered
            self.state.time_in_current_step = 0  # Time elapsed in current step (seconds)
            self.state.sequence_complete = False
            # For overlap mode, track the previous step's output for crossfading
            self.state.prev_step_buffer = None  # Buffer to store previous step's output during crossfade
            self.state.prev_step_index = None
            self.state.crossfade_progress = 0  # Samples rendered in current crossfade
            self.state.last_trigger_value = 0.0  # For trigger edge detection
            self.state.last_reset_value = 0.0  # For reset edge detection

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
        return time_to_samples(total_duration )

    def _do_render(self, num_samples=None, context=None, num_channels=1, **params):
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
        
        # Get interval and swing values
        interval_wave = self.interval_node.render(num_samples_resolved, context, **self.get_params_for_children(params))
        swing_wave = self.swing_node.render(num_samples_resolved, context, **self.get_params_for_children(params))
        # Handle both mono and stereo arrays
        if isinstance(interval_wave, np.ndarray) and interval_wave.ndim > 1:
            interval = float(interval_wave[0, 0]) if interval_wave.size > 0 else 1.0
        else:
            interval = float(interval_wave[0]) if len(interval_wave) > 0 else 1.0
        
        if isinstance(swing_wave, np.ndarray) and swing_wave.ndim > 1:
            swing = float(swing_wave[0, 0]) if swing_wave.size > 0 else 0.0
        else:
            swing = float(swing_wave[0]) if len(swing_wave) > 0 else 0.0
        swing = np.clip(swing, -1.0, 1.0)  # Ensure swing is in valid range
        
        # Render trigger and reset signals if provided
        trigger_indices = []
        reset_indices = []
        
        if self.trigger_node is not None:
            trigger_wave = self.trigger_node.render(num_samples_resolved, context, **self.get_params_for_children(params))
            if len(trigger_wave) < num_samples_resolved:
                trigger_wave = np.pad(trigger_wave, (0, num_samples_resolved - len(trigger_wave)))
            elif len(trigger_wave) > num_samples_resolved:
                trigger_wave = trigger_wave[:num_samples_resolved]
            trigger_indices, self.state.last_trigger_value = detect_triggers(trigger_wave, self.state.last_trigger_value)
        
        if self.reset_node is not None:
            reset_wave = self.reset_node.render(num_samples_resolved, context, **self.get_params_for_children(params))
            if len(reset_wave) < num_samples_resolved:
                reset_wave = np.pad(reset_wave, (0, num_samples_resolved - len(reset_wave)))
            elif len(reset_wave) > num_samples_resolved:
                reset_wave = reset_wave[:num_samples_resolved]
            reset_indices, self.state.last_reset_value = detect_triggers(reset_wave, self.state.last_reset_value)
        
        if len(self.step_nodes) == 0:
            return np.zeros(num_samples_resolved, dtype=np.float32)
        
        output_wave = np.zeros(num_samples_resolved, dtype=np.float32)
        samples_written = 0
        samples_per_step = time_to_samples(interval )
        
        # Track which triggers we've processed this chunk
        processed_triggers = set()
        processed_resets = set()
        
        # Determine if we're using trigger mode (has trigger input) or interval mode
        using_trigger_mode = self.trigger_node is not None
        
        while samples_written < num_samples_resolved:
            # Process reset triggers at the current sample position
            for reset_idx in reset_indices:
                if reset_idx == samples_written and reset_idx not in processed_resets:
                    # Reset to step 0
                    self.state.current_step = 0
                    self.state.next_step_to_trigger = 0
                    self.state.current_repeat = 0
                    self.state.time_in_current_step = 0
                    self.state.sequence_complete = False
                    self.state.prev_step_buffer = None
                    self.state.prev_step_index = None
                    self.state.crossfade_progress = 0
                    processed_resets.add(reset_idx)
            
            # Process step triggers at the current sample position
            for trigger_idx in trigger_indices:
                if trigger_idx == samples_written and trigger_idx not in processed_triggers:
                    # In trigger mode: advance to the next step to trigger
                    self.state.current_step = self.state.next_step_to_trigger
                    self.state.next_step_to_trigger += 1
                    self.state.time_in_current_step = 0
                    self.state.prev_step_buffer = None
                    self.state.crossfade_progress = 0
                    processed_triggers.add(trigger_idx)
                    
                    # Check if we've now completed the current repeat
                    if self.state.next_step_to_trigger >= len(self.step_nodes):
                        self.state.current_repeat += 1
                        if self.state.current_repeat >= self.repeat:
                            # We've completed all repeats - stay at the last step forever
                            self.state.sequence_complete = True
                            self.state.next_step_to_trigger = len(self.step_nodes) - 1
                        else:
                            # Start next repeat from the beginning
                            self.state.next_step_to_trigger = 0
                            self.state.time_in_current_step = 0
                            self.state.prev_step_buffer = None
                            self.state.prev_step_index = None
                            self.state.crossfade_progress = 0
            
            # In interval mode: check if we've completed the current repeat
            if self.state.current_step >= len(self.step_nodes) and not using_trigger_mode:
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
            
            # Determine which step to render
            if using_trigger_mode:
                # In trigger mode: only render if a trigger has happened
                # We've been triggered if next_step_to_trigger > 0 OR if we're on a repeat > 0
                has_been_triggered = (self.state.next_step_to_trigger > 0 or self.state.current_repeat > 0)
                if not has_been_triggered:
                    # No trigger yet - output silence
                    chunk = np.zeros(num_samples_resolved - samples_written, dtype=np.float32)
                    output_wave[samples_written:] = chunk
                    break
                step_to_render = self.state.current_step
            else:
                # In interval mode: render current_step as normal
                step_to_render = self.state.current_step
            
            # In trigger mode, render up to next trigger; in interval mode, calculate based on interval with swing
            if using_trigger_mode:
                # Trigger mode: render up to the next trigger (or reset) position, or to the end if none
                next_trigger_pos = None
                for trigger_idx in trigger_indices:
                    if trigger_idx > samples_written:
                        next_trigger_pos = trigger_idx
                        break
                for reset_idx in reset_indices:
                    if reset_idx > samples_written:
                        if next_trigger_pos is None or reset_idx < next_trigger_pos:
                            next_trigger_pos = reset_idx
                
                if next_trigger_pos is not None:
                    samples_to_render = next_trigger_pos - samples_written
                else:
                    samples_to_render = num_samples_resolved - samples_written
            else:
                # Interval mode: calculate current interval with swing
                # Apply swing to odd steps (step % 2 == 1)
                current_interval = interval
                if self.state.current_step % 2 == 1:
                    # Odd step: apply swing
                    # swing = -1 means shorter (0.5x), swing = 0 means normal (1x), swing = 1 means longer (1.5x)
                    swing_factor = 1.0 + (swing * 0.5)
                    current_interval = interval * swing_factor
                else:
                    # Even step: compensate for previous odd step's swing
                    swing_factor = 1.0 - (swing * 0.5)
                    current_interval = interval * swing_factor
                
                # Calculate how many samples we can render from current step
                time_remaining_in_step = current_interval - self.state.time_in_current_step
                samples_remaining_in_step = time_to_samples(time_remaining_in_step )
                samples_to_render = min(samples_remaining_in_step, num_samples_resolved - samples_written)
            
            if samples_to_render <= 0:
                # Move to next step (only in interval mode)
                if not using_trigger_mode:
                    self.state.current_step += 1
                    self.state.time_in_current_step = 0
                    self.state.prev_step_buffer = None
                    self.state.crossfade_progress = 0
                    continue
                else:
                    # In trigger mode, this shouldn't happen - break to avoid infinite loop
                    break
            
            # Render based on mode
            if self.mode == AutomationMode.STEP:
                chunk = self._render_step_mode(samples_to_render, context, params)
            elif self.mode == AutomationMode.RAMP:
                # For ramp mode, we need current_interval - calculate it if in interval mode
                if not using_trigger_mode:
                    current_interval = interval
                    if self.state.current_step % 2 == 1:
                        swing_factor = 1.0 + (swing * 0.5)
                        current_interval = interval * swing_factor
                    else:
                        swing_factor = 1.0 - (swing * 0.5)
                        current_interval = interval * swing_factor
                else:
                    current_interval = interval  # Use base interval for trigger mode
                chunk = self._render_ramp_mode(samples_to_render, current_interval, context, params)
            elif self.mode == AutomationMode.OVERLAP:
                # For overlap mode, we need current_interval - calculate it if in interval mode
                if not using_trigger_mode:
                    current_interval = interval
                    if self.state.current_step % 2 == 1:
                        swing_factor = 1.0 + (swing * 0.5)
                        current_interval = interval * swing_factor
                    else:
                        swing_factor = 1.0 - (swing * 0.5)
                        current_interval = interval * swing_factor
                else:
                    current_interval = interval  # Use base interval for trigger mode
                chunk = self._render_overlap_mode(samples_to_render, current_interval, context, params)
            else:
                chunk = np.zeros(samples_to_render, dtype=np.float32)
            
            # Add to output
            output_wave[samples_written:samples_written + len(chunk)] = chunk
            samples_written += len(chunk)
            
            # Update time in current step (only in interval mode)
            if not using_trigger_mode:
                self.state.time_in_current_step += len(chunk) / SAMPLE_RATE
                
                # Check if we should move to next step (only in interval mode)
                # Calculate current_interval with swing for the check
                current_interval = interval
                if self.state.current_step % 2 == 1:
                    swing_factor = 1.0 + (swing * 0.5)
                    current_interval = interval * swing_factor
                else:
                    swing_factor = 1.0 - (swing * 0.5)
                    current_interval = interval * swing_factor
                
                if self.state.time_in_current_step >= current_interval:
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
            last_value = get_last_or_default(result, 0)
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
                last_value = get_last_or_default(result, 0)
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
            last_value = get_last_or_default(current_value, 0)
            padding = np.full(num_samples - len(current_value), last_value)
            current_value = np.concatenate([current_value, padding])
        
        if len(next_value) < num_samples:
            last_value = get_last_or_default(next_value, 0)
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
        overlap_samples = time_to_samples(self.overlap_time )
        
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
        samples_into_step = time_to_samples(self.state.time_in_current_step )
        
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
            last_value = get_last_or_default(current_output, 0)
            padding = np.full(num_samples - len(current_output), last_value)
            current_output = np.concatenate([current_output, padding])
        
        if in_crossfade and self.step_nodes[self.state.prev_step_index] is not None:
            # Render previous step
            prev_output = self.step_nodes[self.state.prev_step_index].render(num_samples, context, **self.get_params_for_children(params))
            
            # Handle empty array from previous step
            if len(prev_output) == 0:
                prev_output = np.zeros(num_samples, dtype=np.float32)
            elif len(prev_output) < num_samples:
                last_value = get_last_or_default(prev_output, 0)
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
