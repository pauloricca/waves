from __future__ import annotations
from typing import List, Optional, Union
import numpy as np
from pydantic import ConfigDict
from config import SAMPLE_RATE
from sound_library import get_sound_model
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.wavable_value import WavableValue
from utils import look_for_duration, empty_mono, time_to_samples, samples_to_time


class SequencerModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    interval: WavableValue = 0
    repeat: int = 1
    sequence: Optional[List[Union[BaseNodeModel, str, List[Union[str, BaseNodeModel]], None]]] = None
    chain: Optional[List[Union[str, BaseNodeModel]]] = None
    swing: WavableValue = 0


class SequencerNode(BaseNode):
    def __init__(self, model: SequencerModel, node_id: str, state, do_initialise_state=True):
        super().__init__(model, node_id, state, do_initialise_state)
        self.model = model
        self.sequence = model.sequence
        self.chain = model.chain
        self.repeat = model.repeat
        self.interval_node = self.instantiate_child_node(model.interval, "interval")
        self.swing_node = self.instantiate_child_node(model.swing, "swing")
        
        # Persistent state for realtime playback (survives hot reload)
        if do_initialise_state:
            self.state.current_repeat = 0
            self.state.current_step = 0
            self.state.time_in_current_step = 0  # Time elapsed in current step (seconds)
            self.state.all_active_sounds = []  # List of (sound_node, render_args, sound_duration, samples_rendered, step_index) tuples
            self.state.step_triggered = set()  # Set of (repeat, step) tuples that have been triggered
            self.state.sequence_complete = False  # Flag to indicate when sequence playback is done
            self.state.sound_instance_counter = 0  # Counter for unique sound instances
        
        # Non-persistent attributes
        self._last_chunk_samples = None

    def instantiate_sound_node(
            self,
            sound_model: BaseNodeModel,
            sound_name_with_params: str,
            attribute_name: str,
            attribute_index: str
        ):
        from nodes.node_utils.node_string_parser import instantiate_node_from_string
        return instantiate_node_from_string(sound_name_with_params, self.node_id, attribute_name, attribute_index, sound_model)

    def create_sound_nodes_for_step(self, step_index, **params):
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
        for sound_index, sound in enumerate(sounds_in_step):
            if sound:
                self.state.sound_instance_counter += 1
                attribute_name = f"step_{step_index}_{sound_index}"
                attribute_index = self.state.current_repeat
                if isinstance(sound, str):
                    main_sound_name = sound.split()[0]
                    sound_model = get_sound_model(main_sound_name)
                    sound_node, render_args = self.instantiate_sound_node(sound_model, sound, attribute_name, attribute_index)
                    # Calculate the duration of this sound
                    sound_duration = look_for_duration(sound_model) or 1
                    sound_nodes_data.append((sound_node, render_args, sound_duration, 0, step_index))  # (node, args, duration, samples_rendered, step_index)
                else:
                    sound_node = self.instantiate_child_node(sound, attribute_name, attribute_index)
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
        return time_to_samples(total_duration )

    def _do_render(self, num_samples=None, context=None, num_channels=1, **params):
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
        swing_wave = self.swing_node.render(num_samples_resolved, context, **self.get_params_for_children(params))
        # Use first value if interval/swing is a wave (assume constant for now)
        interval = float(interval_wave[0]) if len(interval_wave) > 0 else 0.0
        swing = float(swing_wave[0]) if len(swing_wave) > 0 else 0.0
        swing = np.clip(swing, -1.0, 1.0)  # Ensure swing is in valid range
        
        sequence = self.sequence or self.chain
        if not sequence:
            return np.zeros(num_samples, dtype=np.float32)
        
        output_wave = np.zeros(num_samples, dtype=np.float32)
        samples_written = 0
        
        while samples_written < num_samples:
            # Check if we've completed all repeats
            if self.state.current_repeat >= self.repeat:
                self.state.sequence_complete = True
            
            # If sequence is complete but we still have active sounds, render them
            if self.state.sequence_complete:
                if len(self.state.all_active_sounds) == 0:
                    # No more sounds to render, we're done - return what we have so far
                    # If we haven't written anything yet, return empty array
                    if samples_written == 0:
                        return empty_mono()
                    # Otherwise return the partial buffer we've filled
                    return output_wave[:samples_written]
                # Continue to render active sounds below
            
            # Check if we've completed the sequence (but haven't finished all repeats)
            elif self.state.current_step >= len(sequence):
                # We've finished the sequence steps - loop back immediately
                self.state.current_repeat += 1
                self.state.current_step = 0
                self.state.time_in_current_step = 0
                # Reset triggered steps for new repeat (but keep active sounds playing)
                self.state.step_triggered = set()
                if self.state.current_repeat >= self.repeat:
                    self.state.sequence_complete = True
                continue
            
            # Define step_key for tracking
            step_key = (self.state.current_repeat, self.state.current_step)
            
            # Check if we need to trigger new sounds for the current step (only if not complete)
            if not self.state.sequence_complete:
                if step_key not in self.state.step_triggered:
                    # Trigger sounds for this step
                    new_sounds = self.create_sound_nodes_for_step(self.state.current_step, **params)
                    self.state.all_active_sounds.extend(new_sounds)
                    self.state.step_triggered.add(step_key)
            
            # Calculate how much time is left in this chunk
            remaining_samples = num_samples - samples_written
            
            if self.sequence and not self.state.sequence_complete:
                # For sequence mode: calculate based on interval with swing
                # Apply swing to odd steps (step % 2 == 1)
                current_interval = interval
                if self.state.current_step % 2 == 1:
                    # Odd step: apply swing
                    # swing = -1 means shorter (0.5x), swing = 0 means normal (1x), swing = 1 means longer (1.5x)
                    swing_factor = 1.0 + (swing * 0.5)
                    current_interval = interval * swing_factor
                else:
                    # Even step: compensate for previous odd step's swing
                    # If swing was positive, this step should be shorter to maintain overall tempo
                    # If swing was negative, this step should be longer
                    swing_factor = 1.0 - (swing * 0.5)
                    current_interval = interval * swing_factor
                
                # We're in a step - use current_interval
                step_remaining_time = current_interval - self.state.time_in_current_step
                step_remaining_samples = time_to_samples(step_remaining_time )
                samples_to_render = min(remaining_samples, step_remaining_samples)
            else:
                # For chain mode or when sequence is complete: render all remaining samples
                samples_to_render = remaining_samples
            # Safeguard: if samples_to_render is 0 due to rounding, advance to next step
            # This can happen when time_in_current_step is very close to interval
            if samples_to_render <= 0:
                if self.sequence and not self.state.sequence_complete:
                    # Calculate current interval with swing
                    current_interval = interval
                    if self.state.current_step % 2 == 1:
                        swing_factor = 1.0 + (swing * 0.5)
                        current_interval = interval * swing_factor
                    else:
                        swing_factor = 1.0 - (swing * 0.5)
                        current_interval = interval * swing_factor
                    
                    if self.state.time_in_current_step >= current_interval - 0.0001:
                        self.state.current_step += 1
                        self.state.time_in_current_step = 0
                        continue
                # Otherwise this shouldn't happen - break to avoid infinite loop
                break
            
            # Render all active sounds and mix them
            step_wave = np.zeros(samples_to_render, dtype=np.float32)
            
            sounds_to_remove = []
            for i, (sound_node, render_args, sound_duration, samples_rendered_so_far, step_idx) in enumerate(self.state.all_active_sounds):
                # Check if this sound has finished playing
                total_sound_samples = time_to_samples(sound_duration )
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
                self.state.all_active_sounds[i] = (sound_node, render_args, sound_duration, samples_rendered_so_far + len(sound_chunk), step_idx)
                
                # Mix into step wave (pad if needed)
                if len(sound_chunk) < len(step_wave):
                    sound_chunk = np.pad(sound_chunk, (0, len(step_wave) - len(sound_chunk)))
                
                step_wave[:len(sound_chunk)] += sound_chunk
            
            # Remove finished sounds (in reverse order to avoid index issues)
            for i in reversed(sounds_to_remove):
                self.state.all_active_sounds.pop(i)
            
            # For chain mode: check if we should advance to next step immediately
            # This prevents gaps between chain items
            if self.chain and not self.state.sequence_complete:
                current_step_sounds = [s for s in self.state.all_active_sounds if s[4] == self.state.current_step]
                if len(current_step_sounds) == 0 and step_key in self.state.step_triggered:
                    # Current step has finished, move to next
                    self.state.current_step += 1
                    self.state.time_in_current_step = 0
                    # Don't continue the loop - we want to trigger the next step in the same chunk
                    # if there are still samples to render
            
            # Add to output
            output_wave[samples_written:samples_written + samples_to_render] = step_wave
            samples_written += samples_to_render
            self.state.time_in_current_step += samples_to_time(samples_to_render)
            
            # Check if we should advance to next step for sequence mode
            if self.sequence and not self.state.sequence_complete:
                # Calculate current interval with swing
                current_interval = interval
                if self.state.current_step % 2 == 1:
                    swing_factor = 1.0 + (swing * 0.5)
                    current_interval = interval * swing_factor
                else:
                    swing_factor = 1.0 - (swing * 0.5)
                    current_interval = interval * swing_factor
                
                if self.state.time_in_current_step >= current_interval:
                    # For sequence mode: advance based on interval timing
                    self.state.current_step += 1
                    self.state.time_in_current_step = 0
        
        return output_wave


SEQUENCER_DEFINITION = NodeDefinition("sequencer", SequencerNode, SequencerModel)
