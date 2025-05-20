from __future__ import annotations
from enum import Enum
import random
from typing import List, Optional
import numpy as np
from pydantic import ConfigDict, field_validator

from config import DO_NORMALISE_EACH_SOUND, ENVELOPE_TYPE, SAMPLE_RATE
from constants import RenderArgs
from models.models import BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.wavable_value import InterpolationTypes, WavableValue, WavableValueNode
from nodes.node_utils.base import BaseNode
from vnoise import Noise

from utils import consume_kwargs


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
    attack: float = 0
    release: float = 0
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
        self.model = model
        self.freq = WavableValueNode(model.freq, model.freq_interpolation) if model.freq else None
        self.amp = WavableValueNode(model.amp, model.amp_interpolation)
        self.partials = [instantiate_node(partial) for partial in model.partials]
        self._phase_acc = 0

    def render(self, num_samples, **kwargs):
        frequency_multiplier, amplitude_multiplier, frequency_override, kwargs_for_children = consume_kwargs(
            kwargs, {RenderArgs.FREQUENCY_MULTIPLIER: 1, RenderArgs.AMPLITUDE_MULTIPLIER: 1, RenderArgs.FREQUENCY: None})


        duration = self.model.duration or (num_samples / SAMPLE_RATE)
        number_of_samples_to_render = int(SAMPLE_RATE * duration)
        release_time = self.model.release * duration
        attack_time = self.model.attack * duration
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)

        total_wave = 0 * t

        if frequency_override:
            frequency = frequency_override
        elif self.freq:
            frequency = self.freq.render(number_of_samples_to_render, **kwargs_for_children)
            if len(frequency) == 1:
                frequency = frequency[0]
        else:
            frequency = 1
        
        frequency *= frequency_multiplier

        amplitude = self.amp.render(number_of_samples_to_render, **kwargs_for_children) * amplitude_multiplier
        osc_type = self.model.type

        if osc_type == OscillatorTypes.NOISE:
            total_wave = amplitude * np.random.normal(0, 1, len(t))
        elif osc_type in [OscillatorTypes.SIN, OscillatorTypes.COS]:
            wave_function = np.sin if osc_type == OscillatorTypes.SIN else np.cos
            # Is frequency variable?
            if(isinstance(frequency, np.ndarray)):
                dt = 1 / SAMPLE_RATE
                # Compute cumulative phase
                phase = 2 * np.pi * np.cumsum(frequency) * dt
                total_wave = amplitude * wave_function(phase[:len(total_wave)])
            else:
                total_wave = amplitude * wave_function(2 * np.pi * frequency * t)
        elif osc_type == OscillatorTypes.SQR:
            # Is frequency variable?
            if(isinstance(frequency, np.ndarray)):
                dt = 1 / SAMPLE_RATE
                # Compute cumulative phase
                phase = 2 * np.pi * np.cumsum(frequency) * dt
                total_wave = amplitude * np.sign(np.sin(phase[:len(total_wave)]))
            else:
                total_wave = amplitude * np.sign(np.sin(2 * np.pi * frequency * t))
        elif osc_type == OscillatorTypes.TRI:
            # Is frequency variable?
            if(isinstance(frequency, np.ndarray)):
                dt = 1 / SAMPLE_RATE
                # Compute cumulative phase
                phase = 2 * np.pi * np.cumsum(frequency) * dt
                total_wave = amplitude * (2 / np.pi) * np.arcsin(np.sin(phase[:len(total_wave)]))
            else:
                total_wave = amplitude * (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * frequency * t))
        elif osc_type == OscillatorTypes.SAW:
            # Is frequency variable?
            if(isinstance(frequency, np.ndarray)):
                dt = 1 / SAMPLE_RATE
                # Compute cumulative phase
                phase = 2 * np.pi * np.cumsum(frequency) * dt
                total_wave = amplitude * (2 / np.pi) * np.arctan(np.tan(phase[:len(total_wave)]))
            else:
                total_wave = amplitude * (2 / np.pi) * np.arctan(np.tan(np.pi * frequency * t))
        elif osc_type == OscillatorTypes.PERLIN:
            noise_function = Noise(self.model.seed or random.randint(0, 10000)).noise1
            perlin_noise = np.array(noise_function(t * self.model.scale))
            total_wave = amplitude * perlin_noise

        # Render and add partials
        if len(self.partials) > 0:
            partials_args = {RenderArgs.FREQUENCY_MULTIPLIER: frequency, RenderArgs.AMPLITUDE_MULTIPLIER: amplitude}
            for partial in self.partials:
                partial_wave = partial.render(number_of_samples_to_render, **partials_args, **kwargs_for_children)
                # Pad the shorter wave to match the length of the longer one
                if len(partial_wave) > len(total_wave):
                    total_wave = np.pad(total_wave, (0, len(partial_wave) - len(total_wave)))
                elif len(partial_wave) < len(total_wave):
                    partial_wave = np.pad(partial_wave, (0, len(total_wave) - len(partial_wave)))
                total_wave += partial_wave

        # Apply envelope
        if release_time > 0:
            if ENVELOPE_TYPE == "linear":
                fade_out = np.linspace(1, 0, int(SAMPLE_RATE * release_time))
            else:
                fade_out = np.exp(-np.linspace(0, 5, int(SAMPLE_RATE * release_time)))
            total_wave[-len(fade_out) :] *= fade_out

        if attack_time > 0:
            if ENVELOPE_TYPE == "linear":
                fade_in = np.linspace(0, 1, int(SAMPLE_RATE * attack_time))
            else:
                fade_in = 1 - np.exp(-np.linspace(0, 5, int(SAMPLE_RATE * attack_time)))
            total_wave[: len(fade_in)] *= fade_in

        if DO_NORMALISE_EACH_SOUND:
            total_wave = np.clip(total_wave, -1, 1)  # Ensure wave is in the range [-1, 1]
            total_wave = total_wave.astype(np.float32)  # Convert to float32 for sounddevice

        # Convert from [-1, 1] to [min, max]
        if self.model.min is not None and self.model.max is not None:
            total_wave = (total_wave + 1) / 2
            total_wave = total_wave * (self.model.max - self.model.min) + self.model.min

        return total_wave
    
OSCILLATOR_DEFINITION = NodeDefinition("osc", OscillatorNode, OscillatorModel)
