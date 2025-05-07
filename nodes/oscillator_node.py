import numpy as np

from config import DO_NORMALISE_EACH_SOUND, ENVELOPE_TYPE, SAMPLE_RATE
from models import OscillatorModel, OscillatorTypes
from nodes.instantiate_node import instantiate_node
from nodes.wavable_value_node import WavableValueNode
from nodes.base_node import BaseNode
from vnoise import Noise

class OscillatorNode(BaseNode):
    def __init__(self, wave_model: OscillatorModel):
        self.wave_model = wave_model
        self.freq = WavableValueNode(wave_model.freq, wave_model.freq_interpolation) if wave_model.freq else None
        self.amp = WavableValueNode(wave_model.amp, wave_model.amp_interpolation)
        self.partials = [instantiate_node(partial) for partial in wave_model.partials]
        self._phase_acc = 0

    def render(self, num_samples, **kwargs):
        frequency_multiplier = kwargs.get("frequency_multiplier", 1)
        amplitude_multiplier = kwargs.get("amplitude_multiplier", 1)
        duration = self.wave_model.duration or (num_samples / SAMPLE_RATE)
        release_time = self.wave_model.release * duration
        attack_time = self.wave_model.attack * duration
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)

        total_wave = 0 * t

        if self.freq:
            frequency = self.freq.render(num_samples)
            if len(frequency) == 1:
                frequency = frequency[0]
        else:
            frequency = 1
        
        frequency *= frequency_multiplier

        amplitude = self.amp.render(num_samples) * amplitude_multiplier
        osc_type = self.wave_model.osc

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
            noise_function = Noise(self.wave_model.seed).noise1
            perlin_noise = np.array(noise_function(t * self.wave_model.scale))
            # print(perlin_noise)
            total_wave = amplitude * perlin_noise

        if len(self.partials) > 0:
            for partial in self.partials:
                partial_wave = partial.render(num_samples, frequency_multiplier=frequency, amplitude_multiplier=amplitude)
                # Pad the shorter wave to match the length of the longer one
                if len(partial_wave) > len(total_wave):
                    total_wave = np.pad(total_wave, (0, len(partial_wave) - len(total_wave)))
                elif len(partial_wave) < len(total_wave):
                    partial_wave = np.pad(partial_wave, (0, len(total_wave) - len(partial_wave)))
                total_wave += partial_wave

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
        if self.wave_model.min is not None and self.wave_model.max is not None:
            total_wave = (total_wave + 1) / 2
            total_wave = total_wave * (self.wave_model.max - self.wave_model.min) + self.wave_model.min

        return total_wave