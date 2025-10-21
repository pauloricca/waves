from __future__ import annotations
from typing import List, Optional, Union
import re
import numpy as np
from pydantic import ConfigDict
from config import SAMPLE_RATE
from sound_library import get_sound_model
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.wavable_value import WavableValue
from utils import look_for_duration


class SequencerModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    interval: WavableValue = 0
    repeat: int = 1
    sequence: Optional[List[Union[BaseNodeModel, str, List[Union[str, BaseNodeModel]], None]]] = None
    chain: Optional[List[Union[str, BaseNodeModel]]] = None


class SequencerNode(BaseNode):
    def __init__(self, model: SequencerModel):
        from nodes.node_utils.instantiate_node import instantiate_node
        from nodes.wavable_value import WavableValueNode, WavableValueModel
        super().__init__(model)
        self.sequence = model.sequence
        self.chain = model.chain
        self.repeat = model.repeat
        
        # Wrap interval in WavableValue
        if isinstance(model.interval, BaseNodeModel):
            self.interval_node = instantiate_node(model.interval)
        else:
            wavable_model = WavableValueModel(value=model.interval)
            self.interval_node = WavableValueNode(wavable_model)
        
        # Realtime state tracking
        self.current_repeat = 0
        self.current_step = 0
        self.time_in_current_step = 0  # Time elapsed in current step (seconds)
        self.all_active_sounds = []  # List of (sound_node, render_args, sound_duration, samples_rendered, step_index) tuples
        self.step_triggered = set()  # Set of (repeat, step) tuples that have been triggered
        self.sequence_complete = False  # Flag to indicate when sequence playback is done
    
    def instantiate_sound_node(self, sound_model: BaseNodeModel, sound_name_with_params, **params):
        from nodes.node_utils.instantiate_node import instantiate_node
        parts = sound_name_with_params.split()
        render_args = {}

        # Parse parameters by separating text from numeric parts
        # e.g., "f440" -> param_name="f", value=440
        #       "amp0.5" -> param_name="amp", value=0.5
        #       "t2" -> param_name="t", value=2
        for param in parts[1:]:
            # Use regex to separate alphabetic prefix from numeric suffix
            match = re.match(r'^([a-zA-Z_]+)([-+]?[0-9]*\.?[0-9]+)$', param)
            if match:
                param_name = match.group(1)
                param_value = float(match.group(2))
                render_args[param_name] = param_value
        
        # Update sound model with render args
        # Skip when sound_model is None (should be handled by instantiate_node)
        if sound_model:
            # Create a copy of the sound model to avoid modifying the original
            sound_model = sound_model.model_copy(deep=True)
            # For each render_arg, update the corresponding attribute in the copied sound_model
            for param_name, param_value in render_args.items():
            # Only set parameters that exist in the model to avoid errors
                if hasattr(sound_model, param_name):
                    setattr(sound_model, param_name, param_value)
        
        return instantiate_node(sound_model), render_args

    def create_sound_nodes_for_step(self, step_index, **params):
        """Create sound nodes for a given step in the sequence."""
        # Get the sounds for this step
        sequence = self.sequence or self.chain
        if step_index >= len(sequence):
            return []
        
        sounds_in_step = sequence[step_index]
        
        # Handle None or empty steps
        if sounds_in_step is None:
            return []
        
        # Ensure sound_names is always a list
        if not isinstance(sounds_in_step, list):
            sounds_in_step = [sounds_in_step]
        
        # Create nodes for each sound in the step
        sound_nodes_data = []
        for sound in sounds_in_step:
            if sound:
                if isinstance(sound, str):
                    main_sound_name = sound.split()[0]
                    sound_model = get_sound_model(main_sound_name)
                    sound_node, render_args = self.instantiate_sound_node(sound_model, sound, **params)
                    # Calculate the duration of this sound
                    sound_duration = look_for_duration(sound_model) or 1
                    sound_nodes_data.append((sound_node, render_args, sound_duration, 0, step_index))  # (node, args, duration, samples_rendered, step_index)
                else:
                    from nodes.node_utils.instantiate_node import instantiate_node
                    sound_node = instantiate_node(sound)
                    sound_duration = look_for_duration(sound) or 1
                    sound_nodes_data.append((sound_node, {}, sound_duration, 0, step_index))
        
        return sound_nodes_data
    
    def _calculate_total_samples(self):
        """Calculate total number of samples for the entire sequence"""
        sequence = self.sequence or self.chain
        if not sequence:
            return 0
        
        total_sequence_duration = 0
        for item in sequence:
            if isinstance(item, str):
                # String reference - look up the sound model
                sound_model = get_sound_model(item.split()[0])
                item_duration = look_for_duration(sound_model) or 1.0
                total_sequence_duration += item_duration
            elif isinstance(item, BaseNodeModel):
                # Inline node model
                item_duration = look_for_duration(item) or 1.0
                total_sequence_duration += item_duration
            elif isinstance(item, list):
                # List items (like [lead f80, thump a0.2]) - find longest duration
                max_duration = 0
                for subitem in item:
                    if isinstance(subitem, str):
                        sound_model = get_sound_model(subitem.split()[0])
                        subitem_duration = look_for_duration(sound_model) or 1.0
                    elif isinstance(subitem, BaseNodeModel):
                        subitem_duration = look_for_duration(subitem) or 1.0
                    else:
                        subitem_duration = 1.0
                    max_duration = max(max_duration, subitem_duration)
                total_sequence_duration += max_duration if max_duration > 0 else 1.0
            else:
                total_sequence_duration += 1.0
        
        total_duration = total_sequence_duration * self.repeat
        return int(total_duration * SAMPLE_RATE)

    def _do_render(self, num_samples=None, context=None, **params):
        # If num_samples is None, render the entire sequence
        if num_samples is None:
            num_samples = self.resolve_num_samples(num_samples)
            if num_samples is None:
                # Calculate total duration based on sequence and repeat
                num_samples = self._calculate_total_samples()
                if num_samples == 0:
                    return np.array([])
                self._last_chunk_samples = num_samples
        
        # Evaluate interval for this chunk
        num_samples_resolved = self.resolve_num_samples(num_samples)
        if num_samples_resolved is None:
            num_samples_resolved = num_samples
        
        interval_wave = self.interval_node.render(num_samples_resolved, context, **self.get_params_for_children(params))
        # Use first value if interval is a wave (assume constant interval for now)
        interval = float(interval_wave[0]) if len(interval_wave) > 0 else 0.0
        
        sequence = self.sequence or self.chain
        if not sequence:
            return np.zeros(num_samples, dtype=np.float32)
        
        output_wave = np.zeros(num_samples, dtype=np.float32)
        samples_written = 0
        
        while samples_written < num_samples:
            # Check if we've completed all repeats
            if self.current_repeat >= self.repeat:
                self.sequence_complete = True
            
            # If sequence is complete but we still have active sounds, render them
            if self.sequence_complete:
                if len(self.all_active_sounds) == 0:
                    # No more sounds to render, we're done - return what we have so far
                    # If we haven't written anything yet, return empty array
                    if samples_written == 0:
                        return np.array([], dtype=np.float32)
                    # Otherwise return the partial buffer we've filled
                    return output_wave[:samples_written]
                # Continue to render active sounds below
            
            # Check if we've completed the sequence (but haven't finished all repeats)
            elif self.current_step >= len(sequence):
                # We've finished the sequence steps - loop back immediately
                self.current_repeat += 1
                self.current_step = 0
                self.time_in_current_step = 0
                # Reset triggered steps for new repeat (but keep active sounds playing)
                self.step_triggered = set()
                if self.current_repeat >= self.repeat:
                    self.sequence_complete = True
                continue
            
            # Define step_key for tracking
            step_key = (self.current_repeat, self.current_step)
            
            # Check if we need to trigger new sounds for the current step (only if not complete)
            if not self.sequence_complete:
                if step_key not in self.step_triggered:
                    # Trigger sounds for this step
                    new_sounds = self.create_sound_nodes_for_step(self.current_step, **params)
                    self.all_active_sounds.extend(new_sounds)
                    self.step_triggered.add(step_key)
            
            # Calculate how much time is left in this chunk
            remaining_samples = num_samples - samples_written
            
            if self.sequence and not self.sequence_complete:
                # For sequence mode: calculate based on interval
                # We're in a step - use interval
                step_remaining_time = interval - self.time_in_current_step
                step_remaining_samples = int(step_remaining_time * SAMPLE_RATE)
                samples_to_render = min(remaining_samples, step_remaining_samples)
            else:
                # For chain mode or when sequence is complete: render all remaining samples
                samples_to_render = remaining_samples
            # Safeguard: if samples_to_render is 0 due to rounding, advance to next step
            # This can happen when time_in_current_step is very close to interval
            if samples_to_render <= 0:
                if self.sequence and not self.sequence_complete and self.time_in_current_step >= interval - 0.0001:
                    self.current_step += 1
                    self.time_in_current_step = 0
                    continue
                # Otherwise this shouldn't happen - break to avoid infinite loop
                break
            
            # Render all active sounds and mix them
            step_wave = np.zeros(samples_to_render, dtype=np.float32)
            
            sounds_to_remove = []
            for i, (sound_node, render_args, sound_duration, samples_rendered_so_far, step_idx) in enumerate(self.all_active_sounds):
                # Check if this sound has finished playing
                total_sound_samples = int(sound_duration * SAMPLE_RATE)
                if samples_rendered_so_far >= total_sound_samples:
                    # Sound has finished, mark for removal
                    sounds_to_remove.append(i)
                    continue
                
                # Calculate how many samples we can still render from this sound
                remaining_sound_samples = total_sound_samples - samples_rendered_so_far
                samples_to_render_from_sound = min(samples_to_render, remaining_sound_samples)
                
                if samples_to_render_from_sound <= 0:
                    continue
                
                # Merge render_args with params - pass all params through
                merged_params = params.copy()
                merged_params.update(render_args)
                
                # Render from current position
                sound_chunk = sound_node.render(samples_to_render_from_sound, context, **merged_params)
                
                # If the sound returns empty array, it's done - mark for removal
                if len(sound_chunk) == 0:
                    sounds_to_remove.append(i)
                    continue
                
                # If the sound returns fewer samples than we asked for, it's finishing
                # Update counter and mark for removal if we've caught up
                if len(sound_chunk) < samples_to_render_from_sound:
                    # Sound is finishing - this is the last chunk from it
                    sounds_to_remove.append(i)
                
                # Update samples rendered counter
                self.all_active_sounds[i] = (sound_node, render_args, sound_duration, samples_rendered_so_far + len(sound_chunk), step_idx)
                
                # Mix into step wave (pad if needed)
                if len(sound_chunk) < len(step_wave):
                    sound_chunk = np.pad(sound_chunk, (0, len(step_wave) - len(sound_chunk)))
                
                step_wave[:len(sound_chunk)] += sound_chunk
            
            # Remove finished sounds (in reverse order to avoid index issues)
            for i in reversed(sounds_to_remove):
                self.all_active_sounds.pop(i)
            
            # For chain mode: check if we should advance to next step immediately
            # This prevents gaps between chain items
            if self.chain and not self.sequence_complete:
                current_step_sounds = [s for s in self.all_active_sounds if s[4] == self.current_step]
                if len(current_step_sounds) == 0 and step_key in self.step_triggered:
                    # Current step has finished, move to next
                    self.current_step += 1
                    self.time_in_current_step = 0
                    # Don't continue the loop - we want to trigger the next step in the same chunk
                    # if there are still samples to render
            
            # Add to output
            output_wave[samples_written:samples_written + samples_to_render] = step_wave
            samples_written += samples_to_render
            self.time_in_current_step += samples_to_render / SAMPLE_RATE
            
            # Check if we should advance to next step for sequence mode
            if self.sequence and not self.sequence_complete and self.time_in_current_step >= interval:
                # For sequence mode: advance based on interval timing
                self.current_step += 1
                self.time_in_current_step = 0
        
        return output_wave


SEQUENCER_DEFINITION = NodeDefinition("sequencer", SequencerNode, SequencerModel)
