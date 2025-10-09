from __future__ import annotations
from enum import Enum
import random
from typing import List, Optional
import numpy as np
from pydantic import ConfigDict, field_validator

from config import DO_NORMALISE_EACH_SOUND, OSC_ENVELOPE_TYPE, SAMPLE_RATE
from constants import RenderArgs
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.wavable_value import InterpolationTypes, WavableValue, wavable_value_node_factory
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from vnoise import Noise

from utils import add_waves, multiply_waves


OSCILLATOR_RENDER_ARGS = [
    RenderArgs.FREQUENCY_MULTIPLIER,
    RenderArgs.AMPLITUDE_MULTIPLIER,
    RenderArgs.FREQUENCY,
]


class OscillatorTypes(str, Enum):
    SIN = "SIN"
    COS = "COS"
    TRI = "TRI"
    SQR = "SQR"
    SAW = "SAW"
    NOISE = "NOISE"
    PERLIN = "PERLIN"
    NONE = "NONE"


class OscillatorModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    type: OscillatorTypes = OscillatorTypes.SIN
    freq: Optional[WavableValue] = None
    freq_interpolation: InterpolationTypes = InterpolationTypes.LINEAR
    amp: WavableValue = 1.0
    amp_interpolation: InterpolationTypes = InterpolationTypes.LINEAR
    attack: float = 0 # Attack time in seconds
    release: float = 0 # Release time in seconds
    partials: List[OscillatorModel] = []
    scale: float = 1.0 # Perlin noise scale
    seed: Optional[float] = None # Perlin noise seed
    min: Optional[float] = None # normalized min value
    max: Optional[float] = None # normalized max value
    
    @field_validator("type", mode="before")
    @classmethod
    def normalize_wave_type(cls, v):
        if v is None:
            return OscillatorTypes.SIN.value
        if isinstance(v, str):
            return v.upper()
        return v


class OscillatorNode(BaseNode):
    def __init__(self, model: OscillatorModel):
        from nodes.node_utils.instantiate_node import instantiate_node
        super().__init__(model)
        self.model = model
        self.freq = wavable_value_node_factory(model.freq, model.freq_interpolation) if model.freq else None
        self.amp = wavable_value_node_factory(model.amp, model.amp_interpolation)
        self.partials = [instantiate_node(partial) for partial in model.partials]
        self.seed = self.model.seed or random.randint(0, 10000)
        self.phase_acc = 0  # Phase accumulator to maintain continuity between render calls
        self.fase_in_multiplier: np.ndarray = None
        self.fase_out_multiplier: np.ndarray = None

    def render(self, num_samples=None, **params):
        super().render(num_samples)
        
        # Resolve num_samples from duration if None
        num_samples = self.resolve_num_samples(num_samples)
        if num_samples is None:
            raise ValueError("Cannot render full signal: oscillator has no duration specified")
        
        # Store original num_samples to check if we need to truncate params
        original_num_samples = num_samples
        
        # Check if we've reached the end of our duration
        if self.duration is not None:
            total_duration_samples = int(self.duration * SAMPLE_RATE)
            if self.number_of_chunks_rendered >= total_duration_samples:
                # We're done, return empty array
                return np.array([], dtype=np.float32)
            
            # Check if this chunk will exceed duration
            samples_remaining = total_duration_samples - self.number_of_chunks_rendered
            if samples_remaining < num_samples:
                # This is the last chunk, render only what's needed
                num_samples = samples_remaining
        
        frequency_multiplier, amplitude_multiplier, frequency_override, params_for_children = self.consume_params(
            params, {RenderArgs.FREQUENCY_MULTIPLIER: 1, RenderArgs.AMPLITUDE_MULTIPLIER: 1, RenderArgs.FREQUENCY: None})

        # If we adjusted num_samples, we need to truncate any array parameters
        if num_samples < original_num_samples:
            if isinstance(frequency_multiplier, np.ndarray):
                frequency_multiplier = frequency_multiplier[:num_samples]
            if isinstance(amplitude_multiplier, np.ndarray):
                amplitude_multiplier = amplitude_multiplier[:num_samples]
            if isinstance(frequency_override, np.ndarray):
                frequency_override = frequency_override[:num_samples]
        
        # These need to be calculated AFTER num_samples is potentially adjusted
        chunk_duration_seconds = num_samples / SAMPLE_RATE
        t = np.linspace(0, chunk_duration_seconds, num_samples, endpoint=False)

        total_wave = np.zeros(num_samples)

        if frequency_override:
            frequency = frequency_override
        elif self.freq:
            frequency = self.freq.render(num_samples, **params_for_children)
            if len(frequency) == 1:
                frequency = frequency[0]
        else:
            frequency = 1
        
        frequency *= frequency_multiplier

        amplitude = self.amp.render(num_samples, **params_for_children) * amplitude_multiplier
        osc_type = self.model.type

        if osc_type == OscillatorTypes.NOISE:
            total_wave = amplitude * np.random.normal(0, 1, len(t))
        elif osc_type in [OscillatorTypes.SIN, OscillatorTypes.COS]:
            wave_function = np.sin if osc_type == OscillatorTypes.SIN else np.cos
            # Is frequency variable?
            if isinstance(frequency, np.ndarray):
                dt = 1 / SAMPLE_RATE
                # Compute phase increment for each sample
                phase_increments = 2 * np.pi * frequency * dt
                # Calculate cumulative phase
                phase = self.phase_acc + np.cumsum(phase_increments)
                total_wave = amplitude * wave_function(phase[:len(total_wave)])
                # Save the last phase for next render
                self.phase_acc = phase[-1]
            else:
                # Calculate phase with accumulated phase offset
                phase = self.phase_acc + 2 * np.pi * frequency * t
                total_wave = amplitude * wave_function(phase)
                # Update phase accumulator for next render
                self.phase_acc = (self.phase_acc + 2 * np.pi * frequency * chunk_duration_seconds) % (2 * np.pi)
        elif osc_type == OscillatorTypes.SQR:
            if isinstance(frequency, np.ndarray):
                dt = 1 / SAMPLE_RATE
                phase_increments = 2 * np.pi * frequency * dt
                phase = self.phase_acc + np.cumsum(phase_increments)
                total_wave = amplitude * np.sign(np.sin(phase[:len(total_wave)]))
                self.phase_acc = phase[-1]
            else:
                phase = self.phase_acc + 2 * np.pi * frequency * t
                total_wave = amplitude * np.sign(np.sin(phase))
                self.phase_acc = (self.phase_acc + 2 * np.pi * frequency * chunk_duration_seconds) % (2 * np.pi)
        elif osc_type == OscillatorTypes.TRI:
            if isinstance(frequency, np.ndarray):
                dt = 1 / SAMPLE_RATE
                phase_increments = 2 * np.pi * frequency * dt
                phase = self.phase_acc + np.cumsum(phase_increments)
                total_wave = amplitude * (2 / np.pi) * np.arcsin(np.sin(phase[:len(total_wave)]))
                self.phase_acc = phase[-1]
            else:
                phase = self.phase_acc + 2 * np.pi * frequency * t
                total_wave = amplitude * (2 / np.pi) * np.arcsin(np.sin(phase))
                self.phase_acc = (self.phase_acc + 2 * np.pi * frequency * chunk_duration_seconds) % (2 * np.pi)
        elif osc_type == OscillatorTypes.SAW:
            if isinstance(frequency, np.ndarray):
                dt = 1 / SAMPLE_RATE
                phase_increments = 2 * np.pi * frequency * dt
                phase = self.phase_acc + np.cumsum(phase_increments)
                total_wave = amplitude * (2 / np.pi) * np.arctan(np.tan(phase[:len(total_wave)] / 2))
                self.phase_acc = phase[-1]
            else:
                phase = self.phase_acc + np.pi * frequency * t
                total_wave = amplitude * (2 / np.pi) * np.arctan(np.tan(phase))
                self.phase_acc = (self.phase_acc + np.pi * frequency * chunk_duration_seconds) % np.pi
        elif osc_type == OscillatorTypes.PERLIN:
            continuous_t = t + self.phase_acc
            self.phase_acc = continuous_t[-1]
            noise_function = Noise(self.seed).noise1
            perlin_noise = np.array(noise_function(continuous_t * self.model.scale))
            total_wave = amplitude * perlin_noise

        # Render and add partials
        if len(self.partials) > 0:
            partials_args = {RenderArgs.FREQUENCY_MULTIPLIER: frequency, RenderArgs.AMPLITUDE_MULTIPLIER: amplitude}
            for partial in self.partials:
                # Pass params_for_children first, then partials_args to override with our local values
                partial_wave = partial.render(num_samples, **params_for_children, **partials_args)
                total_wave = add_waves(total_wave, partial_wave)


        attack_number_of_samples = int(self.model.attack * SAMPLE_RATE)

        if attack_number_of_samples > 0:
            if self.fase_in_multiplier is None:
                if OSC_ENVELOPE_TYPE == "linear":
                    self.fase_in_multiplier = np.linspace(0, 1, attack_number_of_samples)
                else:
                    self.fase_in_multiplier = 1 - np.exp(-np.linspace(0, 5, attack_number_of_samples))
            
            total_wave = multiply_waves(total_wave, self.fase_in_multiplier[:len(total_wave)])
            self.fase_in_multiplier = self.fase_in_multiplier[len(total_wave):]  # Keep the rest for next render

        if self.duration is not None:
            release_number_of_samples = int(self.model.release * SAMPLE_RATE)
            full_number_of_samples = int(self.duration * SAMPLE_RATE)

            if release_number_of_samples > 0:
                if self.fase_out_multiplier is None:
                    if OSC_ENVELOPE_TYPE == "linear":
                        self.fase_out_multiplier = np.linspace(1, 0, release_number_of_samples)
                    else:
                        self.fase_out_multiplier = np.exp(-np.linspace(0, 5, release_number_of_samples))
                
                start_of_release = full_number_of_samples - release_number_of_samples

                if start_of_release < self.number_of_chunks_rendered + num_samples:
                    # We are in the release phase
                    if start_of_release >= self.number_of_chunks_rendered:
                        # Release starts within this chunk
                        # Calculate how much we should pad the fade out, so that it starts at the right place
                        fade_out_padding = start_of_release - self.number_of_chunks_rendered
                        # Pad the left of self.fase_out_multiplier with ones
                        ones_padding = np.ones(fade_out_padding)
                        padded_fade_out = np.concatenate([ones_padding, self.fase_out_multiplier])
                        self.fase_out_multiplier = padded_fade_out
                    
                    # Apply fade out
                    if len(self.fase_out_multiplier) < len(total_wave):
                        # If we don't have enough fade out samples, we pad with ones
                        self.fase_out_multiplier = np.concatenate([self.fase_out_multiplier, np.zeros(len(total_wave) - len(self.fase_out_multiplier))])
                    total_wave = multiply_waves(total_wave, self.fase_out_multiplier[:len(total_wave)])
                    self.fase_out_multiplier = self.fase_out_multiplier[len(total_wave):]  # Keep the rest for next render

        if DO_NORMALISE_EACH_SOUND:
            total_wave = np.clip(total_wave, -1, 1)  # Ensure wave is in the range [-1, 1]
            total_wave = total_wave.astype(np.float32)  # Convert to float32 for sounddevice

        # Convert from [-1, 1] to [min, max] (or [-max, max] if min is None)
        if self.model.max is not None:
            min_range = self.model.min if self.model.min is not None else -self.model.max
            total_wave = (total_wave + 1) / 2
            total_wave = total_wave * (self.model.max - min_range) + min_range

        return total_wave
    
OSCILLATOR_DEFINITION = NodeDefinition("osc", OscillatorNode, OscillatorModel)
