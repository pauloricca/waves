    
from __future__ import annotations
import numpy as np
from config import SAMPLE_RATE
from models.models import SequencerModel
from nodes.base_node import BaseNode
from nodes.instantiate_node import instantiate_node
from sound_library import get_sound_model


class SequencerNode(BaseNode):
    def __init__(self, sequence_model: SequencerModel):
        self.sequence = sequence_model.sequence
        self.chain = sequence_model.chain
        self.interval = sequence_model.interval
        self.repeat = sequence_model.repeat

    def render(self, num_samples):
        generated_waves = {}

        # Get unique sounds (sound names with parameters) in the sequence
        unique_sounds = set()
        for sound_names in (self.sequence or self.chain):
            if isinstance(sound_names, str):
                unique_sounds.add(sound_names)
            elif isinstance(sound_names, list):
                unique_sounds.update(sound_names)

        for sound_names in unique_sounds:
            if sound_names and sound_names not in generated_waves:
                parts = sound_names.split()
                main_sound_name = parts[0]
                params = parts[1:] if len(parts) > 1 else []
                
                sound_model = get_sound_model(main_sound_name)

                for param in params:
                    if param.startswith("f"):
                        print("Setting frequency to", param[1:])
                        sound_model.freq = float(param[1:])
                    elif param.startswith("a"):
                        sound_model.amp *= float(param[1:])
                
                sound_node = instantiate_node(sound_model)
                generated_waves[sound_names] = sound_node.render(num_samples)

        # Create a combined wave based on the sequence
        combined_wave = np.array([], dtype=np.float32)

        max_length = int(SAMPLE_RATE * (self.interval * (len(self.sequence) + 1))) if self.sequence else sum([len(generated_waves[sound_names]) for sound_names in self.chain])
        combined_wave = np.zeros(max_length, dtype=np.float32)

        last_end_idx = 0
        for i, sound_names in enumerate(self.sequence or self.chain):
            # Ensure sound_names is always a list
            if isinstance(sound_names, str):
                sound_names = [sound_names]

            if sound_names:
                sequence_of_waves = [
                    generated_waves[name] if name else [] for name in sound_names
                ]
                max_wave_length = max(len(w) for w in sequence_of_waves)
                overlapping_waves = [
                    np.pad(w, (0, max_wave_length - len(w))) for w in sequence_of_waves
                ]
                wave = sum(overlapping_waves)
            else:
                wave = []

            if self.sequence:
                start_idx = int(SAMPLE_RATE * self.interval * i)
            else:
                start_idx = last_end_idx

            end_idx = start_idx + len(wave)

            if end_idx > len(combined_wave):
                combined_wave = np.pad(combined_wave, (0, end_idx - len(combined_wave)))

            combined_wave[start_idx:end_idx] += wave
            last_end_idx = end_idx
        
        # Repeat the combined wave "repeat" times with "interval" seconds in between
        repeated_wave = np.array([], dtype=np.float32)
        for _ in range(self.repeat):
            repeated_wave = np.concatenate(
                (repeated_wave, combined_wave, np.zeros(int(SAMPLE_RATE * self.interval)))
            )
        combined_wave = repeated_wave

        return combined_wave