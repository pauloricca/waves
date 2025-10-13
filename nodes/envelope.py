from __future__ import annotations
from typing import Union
import numpy as np
from config import SAMPLE_RATE, OSC_ENVELOPE_TYPE
from constants import RenderArgs
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition


class EnvelopeModel(BaseNodeModel):
    attack: Union[float, str] = 0 # length of attack in seconds (or expression)
    decay: Union[float, str] = 0 # length of decay in seconds (or expression)
    sustain: Union[float, str] = 1.0 # sustain level (0 to 1) (or expression)
    release: Union[float, str] = 0 # length of release in seconds (or expression)
    signal: BaseNodeModel = None


class EnvelopeNode(BaseNode):
    def __init__(self, model: EnvelopeModel):
        from nodes.node_utils.instantiate_node import instantiate_node
        super().__init__(model)
        self.model = model
        self.signal_node = instantiate_node(model.signal)
        
        # State tracking for real-time rendering
        self.fade_in_multiplier = None  # Attack envelope
        self.decay_multiplier = None  # Decay envelope
        self.fade_out_multiplier = None  # Release envelope
        self.is_in_decay_phase = False
        self.is_in_sustain_phase = False
        self.is_in_release_phase = False
        self.release_started = False
        self.current_amplitude = 0.0  # Track actual current amplitude for smooth release

    def _do_render(self, num_samples, context=None, **params):
        # Evaluate expression parameters
        attack = self.eval_scalar(self.model.attack, context, **params)
        decay = self.eval_scalar(self.model.decay, context, **params)
        sustain = self.eval_scalar(self.model.sustain, context, **params)
        release = self.eval_scalar(self.model.release, context, **params)
        
        # Check if we should start the release phase
        is_in_sustain = params.get(RenderArgs.IS_IN_SUSTAIN, True)
        
        # Transition from sustain to release
        if not is_in_sustain and not self.is_in_release_phase:
            self.is_in_release_phase = True
            self.release_started = True
        
        # Optimization: If release is complete, don't render the signal at all
        release_len = int(release * SAMPLE_RATE)
        if self.is_in_release_phase and release_len > 0:
            if self.fade_out_multiplier is not None and len(self.fade_out_multiplier) == 0:
                # Release is complete, return empty array immediately
                return np.array([])
        elif self.is_in_release_phase and release_len == 0:
            # No release time, stop immediately
            return np.array([])
        
        # Get the signal from the child node
        signal_wave = self.signal_node.render(num_samples, context, **self.get_params_for_children(params))
        
        # If the signal is exhausted and we haven't started release yet, return empty
        if len(signal_wave) == 0 and not self.is_in_release_phase:
            return np.array([])
        
        # Track which samples have been processed to avoid double-processing
        processed_samples = 0
        
        # Apply attack envelope
        attack_len = int(attack * SAMPLE_RATE)
        if attack_len > 0:
            if self.fade_in_multiplier is None:
                # Create the attack envelope (0 to 1)
                if OSC_ENVELOPE_TYPE == "linear":
                    self.fade_in_multiplier = np.linspace(0, 1, attack_len)
                else:
                    self.fade_in_multiplier = 1 - np.exp(-np.linspace(0, 5, attack_len))
            
            # Apply fade in to the current chunk
            if len(self.fade_in_multiplier) > 0 and len(signal_wave) > 0:
                fade_samples = min(len(self.fade_in_multiplier), len(signal_wave))
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
            
            # If release is already complete, return empty array
            if self.fade_out_multiplier is not None and len(self.fade_out_multiplier) == 0:
                return np.array([])
            
            # Apply fade out to the current chunk
            if self.fade_out_multiplier is not None and len(signal_wave) > 0:
                fade_samples = min(len(self.fade_out_multiplier), len(signal_wave))
                signal_wave[:fade_samples] *= self.fade_out_multiplier[:fade_samples]
                # Keep the rest for next render
                self.fade_out_multiplier = self.fade_out_multiplier[fade_samples:]
                
                # If we've consumed all the release envelope, we're done
                if len(self.fade_out_multiplier) == 0:
                    # Pad the rest with zeros if needed, and this is the last chunk
                    if fade_samples < len(signal_wave):
                        signal_wave[fade_samples:] = 0
                    # Next render will return empty
                    return signal_wave
        elif self.is_in_release_phase and release_len == 0:
            # No release time, stop immediately
            return np.array([])

        return signal_wave


ENVELOPE_DEFINITION = NodeDefinition("envelope", EnvelopeNode, EnvelopeModel)
