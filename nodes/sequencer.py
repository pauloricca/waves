from __future__ import annotations
import math
from typing import List, Optional, Union
import numpy as np
from pydantic import ConfigDict
from config import SAMPLE_RATE
from sound_library import get_sound_model
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.wavable_value import WavableValue
from utils import look_for_duration, empty_mono, time_to_samples, samples_to_time, detect_triggers, to_stereo, to_mono, is_stereo


class SequencerModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    interval: WavableValue = 0
    repeat: int = math.inf
    steps: Optional[List[Union[BaseNodeModel, str, List[Union[str, BaseNodeModel]], None]]] = None
    chain: Optional[List[Union[str, BaseNodeModel]]] = None
    swing: WavableValue = 0
    trigger: Optional[WavableValue] = None  # Optional trigger signal to advance steps
    reset: Optional[WavableValue] = None  # Optional reset signal to reset to step 0


class SequencerNode(BaseNode):
    def __init__(self, model: SequencerModel, node_id: str, state, do_initialise_state=True):
        super().__init__(model, node_id, state, do_initialise_state)
        self.model = model
        self.steps = model.steps
        self.chain = model.chain
        self.repeat = model.repeat
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
            self.state.all_active_sounds = []  # List of (sound_node, render_args, sound_duration, samples_rendered, step_index) tuples
            self.state.step_triggered = set()  # Set of (repeat, step) tuples that have been triggered
            self.state.sequence_complete = False  # Flag to indicate when sequence playback is done
            self.state.sound_instance_counter = 0  # Counter for unique sound instances
            self.state.last_trigger_value = 0.0  # For trigger edge detection
            self.state.last_reset_value = 0.0  # For reset edge detection
        
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
        sequence = self.steps or self.chain
        if step_index >= len(sequence):
            return []
        
        sounds_in_step = sequence[step_index]
        
        # Handle None or empty steps
        if sounds_in_step is None:
            return []
        
        # Ensure sound_names is always a list (split comma-separated strings if present)
        if not isinstance(sounds_in_step, list):
            if isinstance(sounds_in_step, str) and ',' in sounds_in_step:
                sounds_in_step = [s.strip() for s in sounds_in_step.split(',')]
            else:
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
        sequence = self.steps or self.chain
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
        swing_wave = self.swing_node.render(num_samples_resolved, context, **self.get_params_for_children(params))
        
        # Convert control signals to mono if needed
        if is_stereo(interval_wave):
            interval_wave = to_mono(interval_wave)
        if is_stereo(swing_wave):
            swing_wave = to_mono(swing_wave)
        
        # Use first value if interval/swing is a wave (assume constant for now)
        interval = float(interval_wave[0]) if len(interval_wave) > 0 else 0.0
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
        
        sequence = self.steps or self.chain
        if not sequence:
            return np.zeros(num_samples, dtype=np.float32)
        
        # Initialize output wave as mono, will convert to stereo if needed
        output_wave = np.zeros(num_samples, dtype=np.float32)
        output_is_stereo = False  # Track if any child returned stereo
        samples_written = 0
        
        # Track which triggers we've processed this chunk
        processed_triggers = set()
        processed_resets = set()
        
        while samples_written < num_samples:
            # Determine if we're using trigger mode (has trigger input) or interval mode
            using_trigger_mode = self.trigger_node is not None
            
            # Process reset triggers at the current sample position
            for reset_idx in reset_indices:
                if reset_idx == samples_written and reset_idx not in processed_resets:
                    # Reset to step 0
                    self.state.current_step = 0
                    self.state.next_step_to_trigger = 0
                    self.state.current_repeat = 0
                    self.state.time_in_current_step = 0
                    self.state.sequence_complete = False
                    self.state.step_triggered = set()  # Clear triggered steps so they can trigger again
                    processed_resets.add(reset_idx)
            
            # Process step triggers at the current sample position
            for trigger_idx in trigger_indices:
                if trigger_idx == samples_written and trigger_idx not in processed_triggers:
                    # In trigger mode: trigger the next step and advance
                    step_to_trigger = self.state.next_step_to_trigger
                    
                    # Trigger sounds for this step
                    step_key = (self.state.current_repeat, step_to_trigger)
                    if step_key not in self.state.step_triggered:
                        new_sounds = self.create_sound_nodes_for_step(step_to_trigger, **params)
                        self.state.all_active_sounds.extend(new_sounds)
                        self.state.step_triggered.add(step_key)
                    
                    # Advance to next step
                    self.state.next_step_to_trigger += 1
                    self.state.current_step = step_to_trigger  # Update current_step to match what was just triggered
                    self.state.time_in_current_step = 0
                    processed_triggers.add(trigger_idx)
                    
                    # Check if we've now completed the sequence
                    if self.state.next_step_to_trigger >= len(sequence):
                        # We've finished the sequence steps - loop back immediately
                        self.state.current_repeat += 1
                        self.state.next_step_to_trigger = 0
                        self.state.time_in_current_step = 0
                        # Reset triggered steps for new repeat (but keep active sounds playing)
                        self.state.step_triggered = set()
            
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
            
            # In interval mode: check if we've completed the sequence and handle step advancement
            if not using_trigger_mode:
                # Check if we've completed the sequence (but haven't finished all repeats)
                if self.state.current_step >= len(sequence):
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
            
            # Determine if we're using trigger mode (has trigger input) or interval mode
            using_trigger_mode = self.trigger_node is not None
            
            if self.steps and not self.state.sequence_complete and not using_trigger_mode:
                # For sequence mode with interval: calculate based on interval with swing
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
            elif using_trigger_mode:
                # In trigger mode: render up to the next trigger (or reset) position, or to the end if none
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
                    samples_to_render = remaining_samples
            else:
                # For chain mode or when sequence is complete: render all remaining samples
                samples_to_render = remaining_samples
            
            # Safeguard: if samples_to_render is 0 due to rounding, advance to next step
            # This can happen when time_in_current_step is very close to interval
            if samples_to_render <= 0:
                if self.steps and not self.state.sequence_complete and not using_trigger_mode:
                    # Calculate current interval with swing (only for interval mode)
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
            step_is_stereo = False
            
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
                
                # Render from current position (may return mono or stereo)
                sound_chunk = sound_node.render(samples_to_render_from_sound, context, **merged_params)
                
                # If the sound returns empty array, it's done - mark for removal
                if len(sound_chunk) == 0:
                    sounds_to_remove.append(i)
                    continue
                
                # Check if this sound returned stereo
                chunk_is_stereo = is_stereo(sound_chunk)
                if chunk_is_stereo and not step_is_stereo:
                    # First stereo sound - convert step_wave to stereo
                    step_wave = to_stereo(step_wave)
                    step_is_stereo = True
                    # Also convert output_wave to stereo immediately if needed
                    if not output_is_stereo:
                        output_wave = to_stereo(output_wave)
                        output_is_stereo = True
                elif not chunk_is_stereo and step_is_stereo:
                    # This sound is mono but step_wave is stereo - convert sound to stereo
                    sound_chunk = to_stereo(sound_chunk)
                
                # If the sound returns fewer samples than we asked for, it's finishing
                # Update counter and mark for removal if we've caught up
                if len(sound_chunk) < samples_to_render_from_sound:
                    # Sound is finishing - this is the last chunk from it
                    sounds_to_remove.append(i)
                
                # Update samples rendered counter
                self.state.all_active_sounds[i] = (sound_node, render_args, sound_duration, samples_rendered_so_far + len(sound_chunk), step_idx)
                
                # Mix into step wave (pad if needed)
                if len(sound_chunk) < len(step_wave):
                    if step_is_stereo:
                        # Stereo padding
                        sound_chunk = np.pad(sound_chunk, ((0, len(step_wave) - len(sound_chunk)), (0, 0)))
                    else:
                        # Mono padding
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
            
            # Add to output - handle stereo/mono conversion
            if step_is_stereo and not output_is_stereo:
                # First stereo step - convert entire output_wave to stereo (including already written samples)
                output_wave = to_stereo(output_wave)
                output_is_stereo = True
            elif not step_is_stereo and output_is_stereo:
                # This step is mono but output is stereo - convert step to stereo
                step_wave = to_stereo(step_wave)
                step_is_stereo = True
            
            # Now both output_wave and step_wave should have matching dimensionality
            if output_is_stereo:
                output_wave[samples_written:samples_written + samples_to_render, :] = step_wave
            else:
                output_wave[samples_written:samples_written + samples_to_render] = step_wave
            samples_written += samples_to_render
            
            # Only update time tracking if we're in interval mode (not trigger mode)
            if not using_trigger_mode:
                self.state.time_in_current_step += samples_to_time(samples_to_render)
            
            # Check if we should advance to next step for sequence mode (only in interval mode)
            if self.steps and not self.state.sequence_complete and not using_trigger_mode:
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
