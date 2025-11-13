from __future__ import annotations
from typing import Union, Optional
import numpy as np
from config import SAMPLE_RATE, OSC_ENVELOPE_TYPE
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.wavable_value import WavableValue
from utils import empty_mono, time_to_samples, detect_triggers, is_stereo


class EnvelopeModel(BaseNodeModel):
    attack: Union[float, str] = 0 # length of attack in seconds (or expression)
    decay: Union[float, str] = 0 # length of decay in seconds (or expression)
    sustain: Union[float, str] = 1.0 # sustain level (0 to 1) (or expression)
    sustain_duration: Union[float, str] | None = None # how long to hold at sustain level before release (or expression). If None and no gate, release immediately after decay
    release: Union[float, str] = 0 # length of release in seconds (or expression)
    gate: Optional[WavableValue] = None  # gate signal (>= 0.5 = on, < 0.5 = trigger release). If None, use duration-based envelope
    trigger: Optional[WavableValue] = None # trigger signal (0→1 crossing retriggers envelope)
    signal: WavableValue = None # If none, uses a constant signal of 1.0, effectively making it a simple envelope generator
    end: bool | None = None # if True, return empty array after release is complete (signals parent to stop rendering)


class EnvelopeNode(BaseNode):
    def __init__(self, model: EnvelopeModel, node_id: str, state, do_initialise_state=True):
        super().__init__(model, node_id, state, do_initialise_state)
        self.model = model
        self.signal_node = self.instantiate_child_node(model.signal, "signal") if model.signal is not None else None
        self.gate_node = self.instantiate_child_node(model.gate, "gate") if model.gate is not None else None
        self.trigger_node = self.instantiate_child_node(model.trigger, "trigger") if model.trigger is not None else None

        # If no gate or end is defined, set end to True for envelope completion
        if model.end is None:
            # If gate is None (duration-based), default end to True
            # If gate is specified (gate-based), default end to False
            self.end = True if model.gate is None else False
        else:
            self.end = model.end
        
        # Persistent state for real-time rendering (survives hot reload)
        if do_initialise_state:
            self.state.fade_in_multiplier = None  # Attack envelope
            self.state.decay_multiplier = None  # Decay envelope
            self.state.fade_out_multiplier = None  # Release envelope
            self.state.is_in_decay_phase = False
            self.state.is_in_sustain_phase = False
            self.state.is_in_release_phase = False
            self.state.release_started = False
            self.state.current_amplitude = 0.0  # Track actual current amplitude for smooth release
            self.state.previous_gate_state = True  # Track previous gate state for retrigger detection
            self.state.sustain_duration_samples = None  # For duration-based envelopes (drums)
            self.state.last_trigger_value = 0.0  # For trigger edge detection
            self.state.envelope_sample_count = 0  # Track samples within current envelope cycle (resets on retrigger)

    def _do_render(self, num_samples, context=None, **params):
        # Evaluate expression parameters
        attack = self.eval_scalar(self.model.attack, context, **params)
        decay = self.eval_scalar(self.model.decay, context, **params)
        sustain = self.eval_scalar(self.model.sustain, context, **params)
        sustain_duration = self.eval_scalar(self.model.sustain_duration, context, **params) if self.model.sustain_duration is not None else None
        release = self.eval_scalar(self.model.release, context, **params)
        
        # Check for trigger events
        should_retrigger = False
        if self.trigger_node is not None:
            trigger_wave = self.trigger_node.render(num_samples, context, **self.get_params_for_children(params))
            if len(trigger_wave) < num_samples:
                trigger_wave = np.pad(trigger_wave, (0, num_samples - len(trigger_wave)))
            elif len(trigger_wave) > num_samples:
                trigger_wave = trigger_wave[:num_samples]
            trigger_indices, self.state.last_trigger_value = detect_triggers(trigger_wave, self.state.last_trigger_value)
            
            # If any triggers detected, retrigger the envelope
            if len(trigger_indices) > 0:
                should_retrigger = True
        
        # Check gate state to determine if we should start release or retrigger
        if self.gate_node is not None:
            # Use gate parameter - render it to check if gate is open (>= 0.5)
            gate = self.gate_node.render(num_samples, context, **self.get_params_for_children(params))
            
            # Check current gate state - use the LAST sample to determine state
            # This way we track where we end up, not where we started
            current_gate_high = gate[-1] >= 0.5 if isinstance(gate, np.ndarray) else gate >= 0.5
            
            # Detect retrigger: gate went from low to high OR explicit trigger signal
            if (not self.state.previous_gate_state and current_gate_high) or should_retrigger:
                # Retrigger! Reset envelope to attack phase
                # Store the current envelope value to start the new attack from
                retrigger_start_amplitude = self.state.current_amplitude
                
                self.state.fade_in_multiplier = None
                self.state.decay_multiplier = None
                self.state.fade_out_multiplier = None
                self.state.is_in_decay_phase = False
                self.state.is_in_sustain_phase = False
                self.state.is_in_release_phase = False
                self.state.release_started = False
                # Set current_amplitude to the stored value for smooth transition
                self.state.current_amplitude = retrigger_start_amplitude
                self.state.sustain_duration_samples = None  # Reset duration tracking
                self.state.envelope_sample_count = 0  # Reset envelope cycle sample counter
            
            # Update previous gate state
            self.state.previous_gate_state = current_gate_high
            
            # Determine if we should be in sustain or release for THIS chunk
            # Use the last gate value to decide
            is_in_sustain = current_gate_high
        else:
            # No gate node - but check if we should retrigger from trigger signal
            if should_retrigger:
                # Retrigger! Reset envelope to attack phase
                retrigger_start_amplitude = self.state.current_amplitude
                
                self.state.fade_in_multiplier = None
                self.state.decay_multiplier = None
                self.state.fade_out_multiplier = None
                self.state.is_in_decay_phase = False
                self.state.is_in_sustain_phase = False
                self.state.is_in_release_phase = False
                self.state.release_started = False
                self.state.current_amplitude = retrigger_start_amplitude
                self.state.sustain_duration_samples = None  # Reset duration tracking
                self.state.envelope_sample_count = 0  # Reset envelope cycle sample counter
            # No gate parameter - use sustain_duration or duration to calculate when to release
            if sustain_duration is not None:
                # Use explicit sustain_duration parameter (takes precedence over duration)
                # Great for trigger-based envelopes where you want precise control
                if self.state.sustain_duration_samples is None:
                    self.state.sustain_duration_samples = time_to_samples(sustain_duration)
                
                # Check if we've exceeded sustain duration
                sustain_end_sample = time_to_samples(attack ) + time_to_samples(decay ) + self.state.sustain_duration_samples
                is_in_sustain = self.state.envelope_sample_count < sustain_end_sample
            elif self.duration is not None:
                # Calculate sustain duration from total duration: total_duration - attack - decay - release
                # Used when you want to specify total envelope length
                if self.state.sustain_duration_samples is None:
                    attack_samples = time_to_samples(attack )
                    decay_samples = time_to_samples(decay )
                    release_samples = time_to_samples(release )
                    total_duration_samples = time_to_samples(self.duration )
                    self.state.sustain_duration_samples = max(0, total_duration_samples - attack_samples - decay_samples - release_samples)
                
                # Check if we've exceeded sustain duration
                sustain_end_sample = time_to_samples(attack ) + time_to_samples(decay ) + self.state.sustain_duration_samples
                is_in_sustain = self.state.envelope_sample_count < sustain_end_sample
            else:
                # No gate, no duration, no trigger - stay open indefinitely (never release)
                is_in_sustain = True
        
        # Transition from sustain to release
        if not is_in_sustain and not self.state.is_in_release_phase:
            self.state.is_in_release_phase = True
            self.state.release_started = True
        
        # Get the signal from the child node (may be mono or stereo). If no signal is defined, render ones (mono)
        signal_wave = self.signal_node.render(num_samples, context, **self.get_params_for_children(
            params)) if self.signal_node is not None else np.ones(num_samples, dtype=np.float32)
        
        # If signal is exhausted, create silent mono output instead of returning empty
        if len(signal_wave) == 0:
            signal_wave = np.zeros(num_samples, dtype=np.float32)
        
        # Determine if signal is stereo (2D array) - envelope will pass through whatever format the child returns
        is_stereo_signal = is_stereo(signal_wave)
        
        # Track which samples have been processed to avoid double-processing
        processed_samples = 0
        
        # Apply attack envelope
        attack_len = time_to_samples(attack )
        if attack_len > 0:
            if self.state.fade_in_multiplier is None:
                # Create the attack envelope from current_amplitude to 1
                # This prevents clicks when retriggering
                if OSC_ENVELOPE_TYPE == "linear":
                    self.state.fade_in_multiplier = np.linspace(self.state.current_amplitude, 1, attack_len)
                else:
                    # Exponential attack from current_amplitude to 1
                    attack_curve = 1 - np.exp(-np.linspace(0, 5, attack_len))
                    # Scale from [0, 1] to [current_amplitude, 1]
                    self.state.fade_in_multiplier = self.state.current_amplitude + (1 - self.state.current_amplitude) * attack_curve
            
            # Apply fade in to the current chunk
            if len(self.state.fade_in_multiplier) > 0 and len(signal_wave) > processed_samples:
                fade_samples = min(len(self.state.fade_in_multiplier), len(signal_wave) - processed_samples)
                multiplier = self.state.fade_in_multiplier[:fade_samples]
                if is_stereo_signal:
                    multiplier = multiplier[:, np.newaxis]  # Reshape for broadcasting with stereo
                signal_wave[processed_samples:processed_samples + fade_samples] *= multiplier
                # Track the current amplitude
                if fade_samples > 0:
                    self.state.current_amplitude = self.state.fade_in_multiplier[fade_samples - 1]
                processed_samples += fade_samples
                # Keep the rest for next render
                self.state.fade_in_multiplier = self.state.fade_in_multiplier[fade_samples:]
                
                # If attack is complete, move to decay phase
                if len(self.state.fade_in_multiplier) == 0:
                    self.state.is_in_decay_phase = True
                    self.state.current_amplitude = 1.0
        else:
            # No attack, go straight to decay
            self.state.is_in_decay_phase = True
            self.state.current_amplitude = 1.0
        
        # Apply decay envelope
        decay_len = time_to_samples(decay )
        if decay_len > 0 and self.state.is_in_decay_phase and not self.state.is_in_sustain_phase and not self.state.is_in_release_phase:
            if self.state.decay_multiplier is None:
                # Create the decay envelope (1 to sustain level)
                if OSC_ENVELOPE_TYPE == "linear":
                    self.state.decay_multiplier = np.linspace(1, sustain, decay_len)
                else:
                    # Exponential decay from 1 to sustain level
                    decay_curve = np.exp(-np.linspace(0, 5, decay_len))
                    # Scale from [1, ~0] to [1, sustain]
                    self.state.decay_multiplier = sustain + (1 - sustain) * decay_curve
            
            # Apply decay to the current chunk
            if len(self.state.decay_multiplier) > 0 and len(signal_wave) > processed_samples:
                fade_samples = min(len(self.state.decay_multiplier), len(signal_wave) - processed_samples)
                multiplier = self.state.decay_multiplier[:fade_samples]
                if is_stereo_signal:
                    multiplier = multiplier[:, np.newaxis]  # Reshape for broadcasting with stereo
                signal_wave[processed_samples:processed_samples + fade_samples] *= multiplier
                # Track the current amplitude
                if fade_samples > 0:
                    self.state.current_amplitude = self.state.decay_multiplier[fade_samples - 1]
                processed_samples += fade_samples
                # Keep the rest for next render
                self.state.decay_multiplier = self.state.decay_multiplier[fade_samples:]
                
                # If decay is complete, move to sustain phase
                if len(self.state.decay_multiplier) == 0:
                    self.state.is_in_sustain_phase = True
                    self.state.current_amplitude = sustain
        elif self.state.is_in_decay_phase and not self.state.is_in_release_phase:
            # No decay time or decay complete, move to sustain
            self.state.is_in_sustain_phase = True
            self.state.current_amplitude = sustain
        
        # Apply sustain level to remaining unprocessed samples (only if not in release)
        if self.state.is_in_sustain_phase and not self.state.is_in_release_phase and processed_samples < len(signal_wave):
            # Calculate how many samples until release should start (if duration-based envelope)
            if self.duration is not None and not self.gate_node:
                sustain_end_sample = time_to_samples(attack) + time_to_samples(decay) + self.state.sustain_duration_samples
                samples_until_release = sustain_end_sample - self.state.envelope_sample_count
                
                # If release starts within this chunk, only apply sustain up to that point
                if samples_until_release < len(signal_wave) - processed_samples:
                    if samples_until_release > 0:
                        signal_wave[processed_samples:processed_samples + samples_until_release] *= sustain
                    # Trigger transition to release for the remaining samples in this chunk
                    self.state.is_in_release_phase = True
                    self.state.release_started = True
                    self.state.is_in_sustain_phase = False
                    processed_samples += samples_until_release
                else:
                    # Apply sustain to all remaining samples
                    signal_wave[processed_samples:] *= sustain
            else:
                # Gate-based or infinite sustain
                signal_wave[processed_samples:] *= sustain
            # Current amplitude stays at sustain level
            self.state.current_amplitude = sustain
        
        # Apply release envelope
        release_len = time_to_samples(release )
        if release_len > 0 and self.state.is_in_release_phase:
            if self.state.fade_out_multiplier is None and self.state.release_started:
                # Create the release envelope starting from CURRENT amplitude (not sustain)
                # This prevents clicks when release starts during attack or decay
                if OSC_ENVELOPE_TYPE == "linear":
                    self.state.fade_out_multiplier = np.linspace(self.state.current_amplitude, 0, release_len)
                else:
                    # Exponential release from current amplitude to 0
                    release_curve = np.exp(-np.linspace(0, 5, release_len))
                    self.state.fade_out_multiplier = self.state.current_amplitude * release_curve
                self.state.release_started = False
            
            # Apply fade out to the current chunk
            if self.state.fade_out_multiplier is not None and len(signal_wave) > processed_samples:
                fade_samples = min(len(self.state.fade_out_multiplier), len(signal_wave) - processed_samples)
                multiplier = self.state.fade_out_multiplier[:fade_samples]
                if is_stereo_signal:
                    multiplier = multiplier[:, np.newaxis]  # Reshape for broadcasting with stereo
                signal_wave[processed_samples:processed_samples + fade_samples] *= multiplier
                # Track the current amplitude during release
                if fade_samples > 0:
                    self.state.current_amplitude = self.state.fade_out_multiplier[fade_samples - 1]
                processed_samples += fade_samples
                # Keep the rest for next render
                self.state.fade_out_multiplier = self.state.fade_out_multiplier[fade_samples:]
                
                # If we've consumed all the release envelope, output silence (envelope stays ready for retrigger)
                if len(self.state.fade_out_multiplier) == 0:
                    # Pad the rest with zeros
                    if fade_samples < len(signal_wave):
                        signal_wave[fade_samples:] = 0
                    self.state.current_amplitude = 0.0  # Release complete, amplitude is now 0
                    
                    # If end=True, return empty array to signal completion
                    if self.end:
                        return empty_mono()
        elif self.state.is_in_release_phase and release_len == 0:
            # No release time, output silence immediately
            signal_wave[:] = 0
            self.state.current_amplitude = 0.0
            
            # If end=True, return empty array to signal completion
            if self.end:
                return empty_mono()
        
        # Increment envelope sample counter for duration tracking
        self.state.envelope_sample_count += num_samples

        return signal_wave


ENVELOPE_DEFINITION = NodeDefinition("envelope", EnvelopeNode, EnvelopeModel)
