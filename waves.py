import numpy as np
import os
import yaml
import sys
import time
import traceback

from config import *
from models import SequenceModel, SoundLibraryModel, WaveModel, WaveTypes
from nodes import instantiate_node
from utils import play, save

rendered_sounds: dict[np.ndarray] = {}

def main():
    global rendered_sounds

    with open(YAML_FILE) as file:
        raw_data = yaml.safe_load(file)

    sound_library = SoundLibraryModel.model_validate(raw_data)

    if len(sys.argv) < 2:
        print("Usage: python waves.py <sound-name>")
        sys.exit(1)

    sound_name_to_play = sys.argv[1]    
    
    if not sound_name_to_play in sound_library.keys():
        print(f"Error: Sound '{sound_name_to_play}' not found in the sound library.")
        sys.exit(1)
    
    sound_to_play = instantiate_node(sound_library[sound_name_to_play])

    rendering_start_time = time.time()

    rendering_end_time = time.time()

    combined_wave = sound_to_play.render(int(SAMPLE_RATE * sound_library[sound_name_to_play].duration))

    # Normalize the combined wave
    peak = np.max(np.abs(combined_wave))
    combined_wave /= DIVIDE_BY

    save(combined_wave, f"{sound_name_to_play}.wav")

    # if DO_SHUFFLE_WAVES:
        # combined_wave_shuffled = shuffle_wave(combined_wave, 0.1)

        # save(combined_wave_shuffled, "combined-shuffled.wav")

        # combined_wave_reverb = add_convolution(combined_wave_shuffled)

        # normalise_wave(combined_wave_reverb)

        # save(combined_wave_reverb, "combined-shuffled-reverb.wav")

        # combined_wave_super_shuffled = shuffle_wave(combined_wave_reverb, 0.3)

        # save(combined_wave_super_shuffled, "combined-shuffled-super-shuffled.wav")

        # combined_wave_re_shuffled = shuffle_wave(combined_wave_shuffled, 0.3)

        # save(combined_wave_re_shuffled, "combined-shuffled-re-shuffled.wav")

    print(f"Sound length: {len(combined_wave) / SAMPLE_RATE:.2f} seconds")
    print(f"Render time: {rendering_end_time - rendering_start_time:.2f} seconds")
    print ("Peak:", peak)

    wave_to_play = combined_wave

    # wave_to_play = add_delay(wave_to_play, 0.3, 3)
    # wave_to_play = add_low_pass_filter(wave_to_play)

    # wave_to_play = normalise_wave(wave_to_play)
    wave_to_play = np.clip(wave_to_play, -1, 1)
    play(wave_to_play)

if __name__ == "__main__":
    last_modified_time = ""
    while True:
        current_modified_time = os.path.getmtime(YAML_FILE)
        if current_modified_time != last_modified_time:
            last_modified_time = current_modified_time
            try:
                main()
            except Exception as e:
                print(f"Error: {e}")
                traceback.print_exc()
        time.sleep(0.5)