import os
import shutil
import sounddevice as sd
import numpy as np
from scipy.io import wavfile

from config import *
from nodes.node_utils.base_node import BaseNodeModel

def play(wave):
    wave = np.clip(wave, -1, 1)
    sd.play(wave, samplerate=SAMPLE_RATE, blocking=True)


def save(wave, filename):
    # Normalize wave to 16-bit PCM format
    wave = np.clip(wave, -1, 1)  # Ensure wave is in the range [-1, 1]
    wave = wave.astype(np.float32)
    wave_int16 = np.int16(wave * np.iinfo(np.int16).max)
    output_path = os.path.join(os.path.dirname(__file__), OUTPUT_DIR, filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    wavfile.write(output_path, SAMPLE_RATE, wave_int16)
    print(f"Saved {filename}")


def load_wav_file(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    sample_rate, data = wavfile.read(filepath)
    if data.ndim > 1:
        # If stereo, take only one channel
        data = data[:, 0]
    data = data.astype(np.float32) / 32767.0  # Normalize to [-1, 1]
    return data


has_printed_visualisation = False
def visualise_wave(wave, do_normalise = False, replace_previous = False, extra_lines = 0):
    global has_printed_visualisation

    wave = wave[0:BUFFER_SIZE]

    # Get terminal width
    visualisation_width = shutil.get_terminal_size().columns
    visualisation_height_resolution_halved = (VISUALISATION_ROW_HEIGHT * 4) // 2

    # Normalize the waveform to fit within the terminal height (otherwise it will clip, which can be useful too)
    if do_normalise:
        wave = wave / np.max(np.abs(wave))

    scaled_wave = (wave * visualisation_height_resolution_halved).astype(int)

    # Group values in scaled_wave into visualisation_width groups
    group_size = max(1, len(scaled_wave) // visualisation_width)
    grouped_wave = [
        (
            min(scaled_wave[i * group_size : (i + 1) * group_size]),
            max(scaled_wave[i * group_size : (i + 1) * group_size]),
        )
        for i in range(visualisation_width)
    ]

    full_visualisation_height = 2 * (VISUALISATION_ROW_HEIGHT // 2)

    if replace_previous and has_printed_visualisation:
        # Clear the terminal
        for _ in range(full_visualisation_height + extra_lines):
            print("\033[1A", end="\x1b[2K")

    # Create a histogram-like visualization (floor the height to make sure we have an even number of rows)
    for i in range(full_visualisation_height):
        line = ""
        for (minVal, maxVal) in grouped_wave:
            if i < visualisation_height_resolution_halved / 4:
                row_value = visualisation_height_resolution_halved - i * 4
                if maxVal > row_value and i == 0:
                    line += "\033[31m█\033[0m" # Red
                elif maxVal >= row_value:
                    line += "█"
                elif maxVal >= row_value - 1:
                    line += "▆" 
                elif maxVal >= row_value - 2:
                    line += "▄"
                elif maxVal >= row_value - 3:
                    line += "▂"
                else:
                    line += " " 
            else:
                row_value = (1 + i - VISUALISATION_ROW_HEIGHT // 2) * -4
                if minVal >= row_value + 4:
                    line += " "
                elif minVal >= row_value + 3:
                    line += "\033[7m▆\033[0m"  # lower third, inverted (looks like upper third)
                elif minVal >= row_value + 2:
                    line += "\033[7m▄\033[0m"  # lower half, inverted (looks like upper half)
                elif minVal >= row_value + 1:
                    line += "\033[7m▃\033[0m"  # lower two thirds, inverted (looks like upper two thirds)
                elif minVal < row_value and i == full_visualisation_height - 1:
                    line += "\033[31m█\033[0m" # Red
                else:
                    line += "█"
        print(line)

    has_printed_visualisation = True



def look_for_duration(model: BaseNodeModel):
        """
        Recursively looks for the duration attribute in the model or its attributes.
        """
        if hasattr(model, "duration") and model.duration is not None:
            return model.duration
        for attr in model.__dict__.values():
            if isinstance(attr, BaseNodeModel):
                duration = look_for_duration(attr)
                if duration is not None:
                    return duration
        
        return None