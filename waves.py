import numpy as np
import os
import yaml
import sys
import time
import traceback

from config import *
from models import SoundLibraryModel
from nodes.instantiate_node import instantiate_node
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
    rendered_sound = sound_to_play.render(int(SAMPLE_RATE * sound_library[sound_name_to_play].duration))
    rendering_end_time = time.time()

    # Normalize the combined wave
    peak = np.max(np.abs(rendered_sound))
    rendered_sound /= DIVIDE_BY

    save(rendered_sound, f"{sound_name_to_play}.wav")

    print(f"Sound length: {len(rendered_sound) / SAMPLE_RATE:.2f} seconds")
    print(f"Render time: {rendering_end_time - rendering_start_time:.2f} seconds")
    print ("Peak:", peak)

    # rendered_sound = normalise_wave(rendered_sound)
    rendered_sound = np.clip(rendered_sound, -1, 1)
    play(rendered_sound)

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