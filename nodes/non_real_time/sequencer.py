    
from __future__ import annotations
from typing import List, Optional, Union
import numpy as np
from pydantic import ConfigDict
from config import SAMPLE_RATE
from constants import RenderArgs
from nodes.oscillator import OSCILLATOR_RENDER_ARGS
from sound_library import get_sound_model
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from utils import look_for_duration


class SequencerModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    interval: float = 0
    repeat: int = 1
    sequence: Optional[List[Union[BaseNodeModel, str, List[Union[str, BaseNodeModel]], None]]] = None
    chain: Optional[List[str]] = None

    def __init__(self, *args, **params):
        super().__init__(*args, **params)
        self.duration = self.interval * (len(self.sequence) + 1) if self.sequence else 0


class SequencerNode(BaseNode):
    def __init__(self, model: SequencerModel):
        super().__init__(model)
        self.sequence = model.sequence
        self.chain = model.chain
        self.interval = model.interval
        self.repeat = model.repeat
    
    def instantiate_sound_node(self, sound_model: BaseNodeModel, sound_name_with_params, num_samples, **params):
        from nodes.node_utils.instantiate_node import instantiate_node
        parts = sound_name_with_params.split()
        params = parts[1:] if len(parts) > 1 else []
        
        render_args = {}

        for param in params:
            if param.startswith("f"):
                render_args[RenderArgs.FREQUENCY] = float(param[1:])
            elif param.startswith("a"):
                render_args[RenderArgs.AMPLITUDE_MULTIPLIER] = float(param[1:])
        
        return instantiate_node(sound_model)

    def render(self, num_samples, **params):
        super().render(num_samples)
        from nodes.node_utils.instantiate_node import instantiate_node
        generated_waves = {}

        # Get unique sounds (sound names with parameters) in the sequence
        unique_sounds = set()
        for sounds_in_step in (self.sequence or self.chain):
            if isinstance(sounds_in_step, str):
                unique_sounds.add(sounds_in_step)
            elif isinstance(sounds_in_step, list):
                unique_sounds.update(sounds_in_step)

        for sounds_in_step in unique_sounds:
            if sounds_in_step and sounds_in_step not in generated_waves:
                main_sound_name = sounds_in_step.split()[0]
                sound_model = get_sound_model(main_sound_name)
                sound_node = self.instantiate_sound_node(sound_model, sounds_in_step, num_samples, **params)
                number_of_samples_to_render = int(SAMPLE_RATE * (look_for_duration(sound_model) or 1))
                # We were adding **self.get_params_for_children(params, OSCILLATOR_RENDER_ARGS), to the below line, cheeck if this is still needed
                generated_waves[sounds_in_step] = sound_node.render(number_of_samples_to_render, **params)

        # Create a combined wave based on the sequence
        combined_wave = np.array([], dtype=np.float32)

        max_length = int(SAMPLE_RATE * (self.interval * (len(self.sequence) + 1))) if self.sequence else sum([len(generated_waves[sound_names]) for sound_names in self.chain])
        combined_wave = np.zeros(max_length, dtype=np.float32)

        last_end_idx = 0
        for i, sounds_in_step in enumerate(self.sequence or self.chain):
            # Ensure sound_names is always a list
            if not isinstance(sounds_in_step, list):
                sounds_in_step = [sounds_in_step]

            if sounds_in_step:
                step_wave = []

                for sound in sounds_in_step:
                    if sound:
                        if isinstance(sound, str):
                            wave = generated_waves[sound]
                        else:
                            sound_node = instantiate_node(sound)
                            wave = sound_node.render(int(SAMPLE_RATE * (look_for_duration(sound) or 1)), **self.get_params_for_children(params, OSCILLATOR_RENDER_ARGS))
                        if len(step_wave) < len(wave):
                            step_wave = np.pad(step_wave, (0, len(wave) - len(step_wave)))
                        elif len(wave) < len(step_wave):
                            wave = np.pad(wave, (0, len(step_wave) - len(wave)))
                        step_wave = step_wave + wave if len(step_wave) else wave
            else:
                step_wave = []

            if self.sequence:
                start_idx = int(SAMPLE_RATE * self.interval * i)
            else:
                start_idx = last_end_idx

            end_idx = start_idx + len(step_wave)

            if end_idx > len(combined_wave):
                combined_wave = np.pad(combined_wave, (0, end_idx - len(combined_wave)))

            combined_wave[start_idx:end_idx] += step_wave
            last_end_idx = end_idx
        
        # Repeat the combined wave "repeat" times with "interval" seconds in between
        repeated_wave = np.array([], dtype=np.float32)
        for _ in range(self.repeat):
            repeated_wave = np.concatenate(
                (repeated_wave, combined_wave, np.zeros(int(SAMPLE_RATE * self.interval)))
            )
        combined_wave = repeated_wave

        return combined_wave
    
SEQUENCER_DEFINITION = NodeDefinition("sequencer", SequencerNode, SequencerModel)
