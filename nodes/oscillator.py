from __future__ import annotations
from enum import Enum
import random
from typing import Optional, Tuple
import numpy as np
from pydantic import ConfigDict, field_validator

from config import DO_NORMALISE_EACH_SOUND, OSC_ENVELOPE_TYPE, SAMPLE_RATE
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.wavable_value import InterpolationTypes, WavableValue
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from vnoise import Noise

from utils import multiply_waves


class OscillatorTypes(str, Enum):
    SIN = "SIN"
    COS = "COS"
    TRI = "TRI"
    SQR = "SQR"
    SAW = "SAW"
    NOISE = "NOISE"
    PERLIN = "PERLIN"
    WANDER = "WANDER"
    NONE = "NONE"


class OscillatorModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    type: OscillatorTypes = OscillatorTypes.SIN
    freq: Optional[WavableValue] = None
    freq_interpolation: InterpolationTypes = InterpolationTypes.LINEAR
    amp: WavableValue = 1.0
    amp_interpolation: InterpolationTypes = InterpolationTypes.LINEAR
    phase: Optional[WavableValue] = None # Phase modulation (for FM synthesis)
    attack: float = 0 # Attack time in seconds
    release: float = 0 # Release time in seconds
    scale: float = 1.0 # Perlin/wander variation rate (higher = faster changes)
    seed: Optional[float] = None # Perlin/wander noise seed
    range: Optional[Tuple[float, float]] = None # Output range [min, max]
    
    @field_validator("type", mode="before")
    @classmethod
    def normalize_wave_type(cls, v):
        if v is None:
            return OscillatorTypes.SIN.value
        if isinstance(v, str):
            return v.upper()
        return v
    
    @field_validator('range', mode='before')
    @classmethod
    def validate_range(cls, v):
        """Convert various input formats to a 2-element tuple"""
        if v is None:
            return None
        
        # If it's already a tuple or list, validate it has exactly 2 elements
        if isinstance(v, (list, tuple)):
            if len(v) != 2:
                raise ValueError(f"range must have exactly 2 values, got {len(v)}")
            return tuple(v)  # Convert to tuple for consistency
        
        raise ValueError(f"range must be a list or tuple with exactly 2 values")


class OscillatorNode(BaseNode):
    def __init__(self, model: OscillatorModel, node_id: str, state=None, do_initialise_state=True):
        super().__init__(model, node_id, state, do_initialise_state)
        self.model = model
        self.freq = self.instantiate_child_node(model.freq, "freq") if model.freq else None
        self.amp = self.instantiate_child_node(model.amp, "amp")
        self.phase_mod = self.instantiate_child_node(model.phase, "phase_mod") if model.phase else None
        self.seed = self.model.seed or random.randint(0, 10000)
        
        # Persistent state (survives hot reload)
        if do_initialise_state:
            self.state.phase_acc = 0  # Phase accumulator to maintain continuity between render calls
            self.state.wander_position = 0.0  # Current position for wander oscillator
            self.state.wander_velocity = 0.0  # Current velocity for wander oscillator
            self.state.wander_rng = None  # Random number generator for wander (initialized on first use)
        
        self.fase_in_multiplier: np.ndarray = None
        self.fase_out_multiplier: np.ndarray = None


    def _do_render(self, num_samples=None, context=None, **params):
        # Resolve num_samples from duration if None
        num_samples = self.resolve_num_samples(num_samples)
        if num_samples is None:
            raise ValueError("Cannot render full signal: oscillator has no duration specified")
        
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
        
        # These need to be calculated AFTER num_samples is potentially adjusted
        chunk_duration_seconds = num_samples / SAMPLE_RATE
        t = np.linspace(0, chunk_duration_seconds, num_samples, endpoint=False)

        total_wave = np.zeros(num_samples)

        params_for_children = self.get_params_for_children(params)

        if self.freq:
            frequency = self.freq.render(num_samples, context, **params_for_children)
            if len(frequency) == 1:
                frequency = frequency[0]
        else:
            frequency = 1

        amplitude = self.amp.render(num_samples, context, **params_for_children)
        
        # Render phase modulation if provided
        phase_modulation = None
        if self.phase_mod:
            phase_modulation = self.phase_mod.render(num_samples, context, **params_for_children)
            # If phase modulation signal has ended (empty array), treat as no modulation
            if len(phase_modulation) == 0:
                phase_modulation = None
            elif len(phase_modulation) == 1:
                phase_modulation = phase_modulation[0]
        
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
                phase = self.state.phase_acc + np.cumsum(phase_increments)
                # Add phase modulation if provided
                if phase_modulation is not None:
                    phase = phase + phase_modulation
                total_wave = amplitude * wave_function(phase[:len(total_wave)])
                # Save the last phase for next render
                self.state.phase_acc = phase[-1] if phase_modulation is None else (phase[-1] - (phase_modulation[-1] if isinstance(phase_modulation, np.ndarray) else phase_modulation))
            else:
                # Calculate phase with accumulated phase offset
                phase = self.state.phase_acc + 2 * np.pi * frequency * t
                # Add phase modulation if provided
                if phase_modulation is not None:
                    phase = phase + phase_modulation
                total_wave = amplitude * wave_function(phase)
                # Update phase accumulator for next render
                phase_without_mod = self.state.phase_acc + 2 * np.pi * frequency * chunk_duration_seconds
                self.state.phase_acc = phase_without_mod % (2 * np.pi)
        elif osc_type == OscillatorTypes.SQR:
            if isinstance(frequency, np.ndarray):
                dt = 1 / SAMPLE_RATE
                phase_increments = 2 * np.pi * frequency * dt
                phase = self.state.phase_acc + np.cumsum(phase_increments)
                # Add phase modulation if provided
                if phase_modulation is not None:
                    phase = phase + phase_modulation
                total_wave = amplitude * np.sign(np.sin(phase[:len(total_wave)]))
                self.state.phase_acc = phase[-1] if phase_modulation is None else (phase[-1] - (phase_modulation[-1] if isinstance(phase_modulation, np.ndarray) else phase_modulation))
            else:
                phase = self.state.phase_acc + 2 * np.pi * frequency * t
                # Add phase modulation if provided
                if phase_modulation is not None:
                    phase = phase + phase_modulation
                total_wave = amplitude * np.sign(np.sin(phase))
                # Update phase accumulator for next render
                phase_without_mod = self.state.phase_acc + 2 * np.pi * frequency * chunk_duration_seconds
                self.state.phase_acc = phase_without_mod % (2 * np.pi)
        elif osc_type == OscillatorTypes.TRI:
            if isinstance(frequency, np.ndarray):
                dt = 1 / SAMPLE_RATE
                phase_increments = 2 * np.pi * frequency * dt
                phase = self.state.phase_acc + np.cumsum(phase_increments)
                # Add phase modulation if provided
                if phase_modulation is not None:
                    phase = phase + phase_modulation
                total_wave = amplitude * (2 / np.pi) * np.arcsin(np.sin(phase[:len(total_wave)]))
                self.state.phase_acc = phase[-1] if phase_modulation is None else (phase[-1] - (phase_modulation[-1] if isinstance(phase_modulation, np.ndarray) else phase_modulation))
            else:
                phase = self.state.phase_acc + 2 * np.pi * frequency * t
                # Add phase modulation if provided
                if phase_modulation is not None:
                    phase = phase + phase_modulation
                total_wave = amplitude * (2 / np.pi) * np.arcsin(np.sin(phase))
                # Update phase accumulator for next render
                phase_without_mod = self.state.phase_acc + 2 * np.pi * frequency * chunk_duration_seconds
                self.state.phase_acc = phase_without_mod % (2 * np.pi)
        elif osc_type == OscillatorTypes.SAW:
            if isinstance(frequency, np.ndarray):
                dt = 1 / SAMPLE_RATE
                phase_increments = 2 * np.pi * frequency * dt
                phase = self.state.phase_acc + np.cumsum(phase_increments)
                # Add phase modulation if provided
                if phase_modulation is not None:
                    phase = phase + phase_modulation
                total_wave = amplitude * (2 / np.pi) * np.arctan(np.tan(phase[:len(total_wave)] / 2))
                self.state.phase_acc = phase[-1] if phase_modulation is None else (phase[-1] - (phase_modulation[-1] if isinstance(phase_modulation, np.ndarray) else phase_modulation))
            else:
                phase = self.state.phase_acc + np.pi * frequency * t
                # Add phase modulation if provided
                if phase_modulation is not None:
                    phase = phase + phase_modulation
                total_wave = amplitude * (2 / np.pi) * np.arctan(np.tan(phase))
                # Update phase accumulator for next render
                phase_without_mod = self.state.phase_acc + np.pi * frequency * chunk_duration_seconds
                self.state.phase_acc = phase_without_mod % np.pi
        elif osc_type == OscillatorTypes.PERLIN:
            continuous_t = t + self.state.phase_acc
            self.state.phase_acc = continuous_t[-1]
            noise_function = Noise(self.seed).noise1
            perlin_noise = np.array(noise_function(continuous_t * self.model.scale))
            total_wave = amplitude * perlin_noise
        elif osc_type == OscillatorTypes.WANDER:
            # Fast random walk LFO - smooth wandering motion
            # scale controls variation rate: higher = faster/more erratic changes (like Perlin)
            if self.state.wander_rng is None:
                self.state.wander_rng = np.random.RandomState(int(self.seed))
            
            # Generate wander wave using smooth random walk
            wander_wave = np.zeros(num_samples, dtype=np.float32)
            
            # Variation rate - higher scale = faster changes (matches Perlin behavior)
            variation_rate = max(0.001, self.model.scale)
            
            for i in range(num_samples):
                # Add random acceleration (changes velocity)
                acceleration = self.state.wander_rng.randn() * variation_rate
                self.state.wander_velocity += acceleration
                
                # Apply damping to velocity to keep it bounded
                damping = 0.95
                self.state.wander_velocity *= damping
                
                # Update position
                self.state.wander_position += self.state.wander_velocity / SAMPLE_RATE
                
                # Soft bounds to keep the output roughly in [-1, 1]
                # Using tanh for smooth bouncing at boundaries
                wander_wave[i] = np.tanh(self.state.wander_position)
            
            total_wave = amplitude * wander_wave

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

        # Convert from [-1, 1] to [min, max] using range parameter
        if self.model.range is not None:
            min_val, max_val = self.model.range
            total_wave = (total_wave + 1) / 2  # Convert from [-1, 1] to [0, 1]
            total_wave = total_wave * (max_val - min_val) + min_val  # Scale to [min, max]

        return total_wave
    
OSCILLATOR_DEFINITION = NodeDefinition("osc", OscillatorNode, OscillatorModel)
