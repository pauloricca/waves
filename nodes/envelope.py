from __future__ import annotations
import numpy as np
from config import SAMPLE_RATE, OSC_ENVELOPE_TYPE
from constants import RenderArgs
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition


class EnvelopeModel(BaseNodeModel):
    attack: float = 0 # length of attack in seconds
    release: float = 0 # length of release in seconds
    signal: BaseNodeModel = None


class EnvelopeNode(BaseNode):
    def __init__(self, model: EnvelopeModel):
        from nodes.node_utils.instantiate_node import instantiate_node
        super().__init__(model)
        self.model = model
        self.signal_node = instantiate_node(model.signal)
        
        # State tracking for real-time rendering
        self.fade_in_multiplier = None  # Attack envelope
        self.fade_out_multiplier = None  # Release envelope
        self.is_in_release_phase = False
        self.release_started = False

    def render(self, num_samples, **params):
        super().render(num_samples)
        
        # Check if we should start the release phase
        is_in_sustain = params.get(RenderArgs.IS_IN_SUSTAIN, True)
        
        # Transition from sustain to release
        if not is_in_sustain and not self.is_in_release_phase:
            self.is_in_release_phase = True
            self.release_started = True
        
        # Get the signal from the child node
        signal_wave = self.signal_node.render(num_samples, **self.get_params_for_children(params))
        
        # If the signal is exhausted and we haven't started release yet, return empty
        if len(signal_wave) == 0 and not self.is_in_release_phase:
            return np.array([])
        
        # Apply attack envelope
        attack_len = int(self.model.attack * SAMPLE_RATE)
        if attack_len > 0:
            if self.fade_in_multiplier is None:
                # Create the attack envelope
                if OSC_ENVELOPE_TYPE == "linear":
                    self.fade_in_multiplier = np.linspace(0, 1, attack_len)
                else:
                    self.fade_in_multiplier = 1 - np.exp(-np.linspace(0, 5, attack_len))
            
            # Apply fade in to the current chunk
            if len(self.fade_in_multiplier) > 0 and len(signal_wave) > 0:
                fade_samples = min(len(self.fade_in_multiplier), len(signal_wave))
                signal_wave[:fade_samples] *= self.fade_in_multiplier[:fade_samples]
                # Keep the rest for next render
                self.fade_in_multiplier = self.fade_in_multiplier[fade_samples:]
        
        # Apply release envelope
        release_len = int(self.model.release * SAMPLE_RATE)
        if release_len > 0 and self.is_in_release_phase:
            if self.fade_out_multiplier is None and self.release_started:
                # Create the release envelope
                if OSC_ENVELOPE_TYPE == "linear":
                    self.fade_out_multiplier = np.linspace(1, 0, release_len)
                else:
                    self.fade_out_multiplier = np.exp(-np.linspace(0, 5, release_len))
                self.release_started = False
            
            # Apply fade out to the current chunk
            if self.fade_out_multiplier is not None and len(signal_wave) > 0:
                fade_samples = min(len(self.fade_out_multiplier), len(signal_wave))
                signal_wave[:fade_samples] *= self.fade_out_multiplier[:fade_samples]
                # Keep the rest for next render
                self.fade_out_multiplier = self.fade_out_multiplier[fade_samples:]
                
                # If we've consumed all the release envelope, we're done
                if len(self.fade_out_multiplier) == 0:
                    # Return the current chunk, but next time return empty
                    # We need to pad with zeros if we still have samples to fill
                    if fade_samples < len(signal_wave):
                        signal_wave[fade_samples:] = 0
                    return signal_wave
            elif self.fade_out_multiplier is not None and len(self.fade_out_multiplier) == 0:
                # Release is complete, return empty array to signal we're done
                return np.array([])

        return signal_wave


ENVELOPE_DEFINITION = NodeDefinition("envelope", EnvelopeNode, EnvelopeModel)
