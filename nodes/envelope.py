from __future__ import annotations
from typing import Union, Optional
import numpy as np
from config import SAMPLE_RATE, OSC_ENVELOPE_TYPE
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.wavable_value import WavableValue, wavable_value_node_factory


class EnvelopeModel(BaseNodeModel):
    attack: Union[float, str] = 0 # length of attack in seconds (or expression)
    decay: Union[float, str] = 0 # length of decay in seconds (or expression)
    sustain: Union[float, str] = 1.0 # sustain level (0 to 1) (or expression)
    release: Union[float, str] = 0 # length of release in seconds (or expression)
    gate: Optional[WavableValue] = None # gate signal (>= 0.5 = on, < 0.5 = trigger release)
    signal: BaseNodeModel = None
    end: bool = False # if True, return empty array after release is complete (signals parent to stop rendering)


class EnvelopeNode(BaseNode):
    def __init__(self, model: EnvelopeModel):
        from nodes.node_utils.instantiate_node import instantiate_node
        super().__init__(model)
        self.model = model
        self.signal_node = instantiate_node(model.signal)
        self.gate_node = wavable_value_node_factory(model.gate) if model.gate is not None else None
        
        # State tracking for real-time rendering
        self.fade_in_multiplier = None  # Attack envelope
        self.decay_multiplier = None  # Decay envelope
        self.fade_out_multiplier = None  # Release envelope
        self.is_in_decay_phase = False
        self.is_in_sustain_phase = False
        self.is_in_release_phase = False
        self.release_started = False
        self.current_amplitude = 0.0  # Track actual current amplitude for smooth release
        self.previous_gate_state = True  # Track previous gate state for retrigger detection
        self.sustain_duration_samples = None  # For duration-based envelopes (drums)

    def _do_render(self, num_samples, context=None, **params):
        # Evaluate expression parameters
        attack = self.eval_scalar(self.model.attack, context, **params)
        decay = self.eval_scalar(self.model.decay, context, **params)
        sustain = self.eval_scalar(self.model.sustain, context, **params)
        release = self.eval_scalar(self.model.release, context, **params)
        
        # Check gate state to determine if we should start release or retrigger
        if self.gate_node is not None:
            # Use gate parameter - render it to check if gate is open (>= 0.5)
            gate = self.gate_node.render(num_samples, context, **self.get_params_for_children(params))
            
            # Check current gate state - use the LAST sample to determine state
            # This way we track where we end up, not where we started
            current_gate_high = gate[-1] >= 0.5 if isinstance(gate, np.ndarray) else gate >= 0.5
            
            # Detect retrigger: gate went from low to high  
            if not self.previous_gate_state and current_gate_high:
                # Retrigger! Reset envelope to attack phase
                # Store the current envelope value to start the new attack from
                retrigger_start_amplitude = self.current_amplitude
                
                self.fade_in_multiplier = None
                self.decay_multiplier = None
                self.fade_out_multiplier = None
                self.is_in_decay_phase = False
                self.is_in_sustain_phase = False
                self.is_in_release_phase = False
                self.release_started = False
                # Set current_amplitude to the stored value for smooth transition
                self.current_amplitude = retrigger_start_amplitude
                self.sustain_duration_samples = None  # Reset duration tracking
            
            # Update previous gate state
            self.previous_gate_state = current_gate_high
            
            # Determine if we should be in sustain or release for THIS chunk
            # Use the last gate value to decide
            is_in_sustain = current_gate_high
        else:
            # No gate parameter - use duration to calculate when to release (for drums)
            if self.duration is not None:
                # Calculate sustain duration: total_duration - attack - decay - release
                if self.sustain_duration_samples is None:
                    attack_samples = int(attack * SAMPLE_RATE)
                    decay_samples = int(decay * SAMPLE_RATE)
                    release_samples = int(release * SAMPLE_RATE)
                    total_duration_samples = int(self.duration * SAMPLE_RATE)
                    self.sustain_duration_samples = max(0, total_duration_samples - attack_samples - decay_samples - release_samples)
                
                # Check if we've exceeded sustain duration
                sustain_end_sample = int(attack * SAMPLE_RATE) + int(decay * SAMPLE_RATE) + self.sustain_duration_samples
                is_in_sustain = self.number_of_chunks_rendered < sustain_end_sample
            else:
                # No gate and no duration - stay open indefinitely (never release)
                is_in_sustain = True
        
        # Transition from sustain to release
        if not is_in_sustain and not self.is_in_release_phase:
            self.is_in_release_phase = True
            self.release_started = True
        
        # Get the signal from the child node
        signal_wave = self.signal_node.render(num_samples, context, **self.get_params_for_children(params))
        
        # If signal is exhausted, create silent output instead of returning empty
        if len(signal_wave) == 0:
            signal_wave = np.zeros(num_samples, dtype=np.float32)
        
        # Track which samples have been processed to avoid double-processing
        processed_samples = 0
        
        # Apply attack envelope
        attack_len = int(attack * SAMPLE_RATE)
        if attack_len > 0:
            if self.fade_in_multiplier is None:
                # Create the attack envelope from current_amplitude to 1
                # This prevents clicks when retriggering
                if OSC_ENVELOPE_TYPE == "linear":
                    self.fade_in_multiplier = np.linspace(self.current_amplitude, 1, attack_len)
                else:
                    # Exponential attack from current_amplitude to 1
                    attack_curve = 1 - np.exp(-np.linspace(0, 5, attack_len))
                    # Scale from [0, 1] to [current_amplitude, 1]
                    self.fade_in_multiplier = self.current_amplitude + (1 - self.current_amplitude) * attack_curve
            
            # Apply fade in to the current chunk
            if len(self.fade_in_multiplier) > 0 and len(signal_wave) > processed_samples:
                fade_samples = min(len(self.fade_in_multiplier), len(signal_wave) - processed_samples)
                signal_wave[processed_samples:processed_samples + fade_samples] *= self.fade_in_multiplier[:fade_samples]
                # Track the current amplitude
                if fade_samples > 0:
                    self.current_amplitude = self.fade_in_multiplier[fade_samples - 1]
                processed_samples += fade_samples
                # Keep the rest for next render
                self.fade_in_multiplier = self.fade_in_multiplier[fade_samples:]
                
                # If attack is complete, move to decay phase
                if len(self.fade_in_multiplier) == 0:
                    self.is_in_decay_phase = True
                    self.current_amplitude = 1.0
        else:
            # No attack, go straight to decay
            self.is_in_decay_phase = True
            self.current_amplitude = 1.0
        
        # Apply decay envelope
        decay_len = int(decay * SAMPLE_RATE)
        if decay_len > 0 and self.is_in_decay_phase and not self.is_in_sustain_phase and not self.is_in_release_phase:
            if self.decay_multiplier is None:
                # Create the decay envelope (1 to sustain level)
                if OSC_ENVELOPE_TYPE == "linear":
                    self.decay_multiplier = np.linspace(1, sustain, decay_len)
                else:
                    # Exponential decay from 1 to sustain level
                    decay_curve = np.exp(-np.linspace(0, 5, decay_len))
                    # Scale from [1, ~0] to [1, sustain]
                    self.decay_multiplier = sustain + (1 - sustain) * decay_curve
            
            # Apply decay to the current chunk
            if len(self.decay_multiplier) > 0 and len(signal_wave) > processed_samples:
                fade_samples = min(len(self.decay_multiplier), len(signal_wave) - processed_samples)
                signal_wave[processed_samples:processed_samples + fade_samples] *= self.decay_multiplier[:fade_samples]
                # Track the current amplitude
                if fade_samples > 0:
                    self.current_amplitude = self.decay_multiplier[fade_samples - 1]
                processed_samples += fade_samples
                # Keep the rest for next render
                self.decay_multiplier = self.decay_multiplier[fade_samples:]
                
                # If decay is complete, move to sustain phase
                if len(self.decay_multiplier) == 0:
                    self.is_in_sustain_phase = True
                    self.current_amplitude = sustain
        elif self.is_in_decay_phase and not self.is_in_release_phase:
            # No decay time or decay complete, move to sustain
            self.is_in_sustain_phase = True
            self.current_amplitude = sustain
        
        # Apply sustain level to remaining unprocessed samples
        if self.is_in_sustain_phase and not self.is_in_release_phase and processed_samples < len(signal_wave):
            signal_wave[processed_samples:] *= sustain
            # Current amplitude stays at sustain level
            self.current_amplitude = sustain
        
        # Apply release envelope
        release_len = int(release * SAMPLE_RATE)
        if release_len > 0 and self.is_in_release_phase:
            if self.fade_out_multiplier is None and self.release_started:
                # Create the release envelope starting from CURRENT amplitude (not sustain)
                # This prevents clicks when release starts during attack or decay
                if OSC_ENVELOPE_TYPE == "linear":
                    self.fade_out_multiplier = np.linspace(self.current_amplitude, 0, release_len)
                else:
                    # Exponential release from current amplitude to 0
                    release_curve = np.exp(-np.linspace(0, 5, release_len))
                    self.fade_out_multiplier = self.current_amplitude * release_curve
                self.release_started = False
            
            # Apply fade out to the current chunk
            if self.fade_out_multiplier is not None and len(signal_wave) > processed_samples:
                fade_samples = min(len(self.fade_out_multiplier), len(signal_wave) - processed_samples)
                signal_wave[processed_samples:processed_samples + fade_samples] *= self.fade_out_multiplier[:fade_samples]
                # Track the current amplitude during release
                if fade_samples > 0:
                    self.current_amplitude = self.fade_out_multiplier[fade_samples - 1]
                processed_samples += fade_samples
                # Keep the rest for next render
                self.fade_out_multiplier = self.fade_out_multiplier[fade_samples:]
                
                # If we've consumed all the release envelope, output silence (envelope stays ready for retrigger)
                if len(self.fade_out_multiplier) == 0:
                    # Pad the rest with zeros
                    if fade_samples < len(signal_wave):
                        signal_wave[fade_samples:] = 0
                    self.current_amplitude = 0.0  # Release complete, amplitude is now 0
                    
                    # If end=True, return empty array to signal completion
                    if self.model.end:
                        return np.array([], dtype=np.float32)
        elif self.is_in_release_phase and release_len == 0:
            # No release time, output silence immediately
            signal_wave[:] = 0
            self.current_amplitude = 0.0
            
            # If end=True, return empty array to signal completion
            if self.model.end:
                return np.array([], dtype=np.float32)

        return signal_wave


ENVELOPE_DEFINITION = NodeDefinition("envelope", EnvelopeNode, EnvelopeModel)
