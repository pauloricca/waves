import os
import shutil
import sounddevice as sd
import numpy as np
from scipy.io import wavfile
from scipy.interpolate import PchipInterpolator

from config import *
from models.models import BaseNodeModel

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
    visualise_wave(wave)
    print(f"Saved {filename}")

def load_wav_file(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    sample_rate, data = wavfile.read(filepath)
    if data.ndim > 1:
        # If stereo, take only one channel
        data = data[:, 0]
    data = data.astype(np.float32) / 32767.0  # Normalize to [-1, 1]
    return data

def visualise_wave(wave, do_normalise = False):
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

    # Create a histogram-like visualization (floor the height to make sure we have an even number of rows)
    for i in range(2 * (VISUALISATION_ROW_HEIGHT // 2)):
        line = ""
        for (minVal, maxVal) in grouped_wave:
            if i < visualisation_height_resolution_halved / 4:
                row_value = visualisation_height_resolution_halved - i * 4
                if maxVal >= row_value:
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
                else:
                    line += "█"
        print(line)

# Interpolates a list of values or a list of lists with relative positions
def interpolate_values(values, num_samples, interpolation_type):
    from nodes.wavable_value import InterpolationTypes
    # If all the values apart from the first and last are lists, we assume they are relative positions
    # It's ok for the first and last not to be lists, as we can assume 0 and 1 positions
    if all(isinstance(v, list) and len(v) == 2 for v in values[1:-1]):        
        # Assume 0 and 1 positions for the first and last values
        if not isinstance(values[0], list):
            values[0] = [values[0], 0]
        if not isinstance(values[-1], list):
            values[-1] = [values[-1], 1]

        # If the first value is not in position 0, we add an extra value at the beginning
        if values[0][1] != 0:
            values = [[values[0][0], 0]] + values
        # If the last value is not in position 1, we add an extra value at the end
        if values[-1][1] != 1:
            values = values + [[values[-1][0], 1]]

        # Handle list of lists with relative positions
        positions = [v[1] for v in values]
        values = [v[0] for v in values]
        if interpolation_type == InterpolationTypes.STEP.value:
            # Step interpolation
            positions = np.array(positions)
            values = np.array(values)
            interpolated_values = np.zeros(num_samples)
            for i in range(len(positions) - 1):
                start = int(positions[i] * num_samples)
                end = int(positions[i + 1] * num_samples)
                interpolated_values[start:end] = values[i]
        elif interpolation_type == InterpolationTypes.SMOOTH.value:
            # Smooth interpolation with a tighter curve
            positions = np.array(positions)
            values = np.array(values)
            x = np.linspace(0, 1, num_samples)
            interpolated_values = np.interp(x, positions, values)
            pchip = PchipInterpolator(positions, values)
            interpolated_values = pchip(x)
        else:
            # Linear interpolation
            positions = np.array(positions)
            values = np.array(values)
            interpolated_values = np.zeros(num_samples)
            for i in range(len(positions) - 1):
                start = int(positions[i] * num_samples)
                end = int(positions[i + 1] * num_samples)
                x = np.linspace(0, 1, end - start)
                interpolated_values[start:end] = np.interp(x, [0, 1], [values[i], values[i + 1]])
    elif len(values) > 1:
        # Handle simple list of values
        if interpolation_type == InterpolationTypes.STEP.value:
            # Step interpolation
            interpolated_values = np.repeat(values, num_samples // len(values))
        elif interpolation_type == InterpolationTypes.SMOOTH.value:
            # Smooth interpolation with a tighter curve
            positions = np.linspace(0, 1, len(values))
            pchip = PchipInterpolator(positions, values)
            x = np.linspace(0, 1, num_samples)
            interpolated_values = pchip(x)
        else:
            # Linear interpolation
            x = np.linspace(0, 1, len(values))
            x_interp = np.linspace(0, 1, num_samples)
            interpolated_values = np.interp(x_interp, x, values)
    else:
        interpolated_values = values[0]
    return interpolated_values

def consume_kwargs(kwargs: dict, keys_and_default_values: dict):
    """
    Extracts keys from `kwargs` using defaults from `defaults`, without mutating the original.
    Returns a tuple of values (in the order of keys_and_default_values) and a new kwargs dict
    with those keys removed.
    """
    remaining = kwargs.copy()
    values = []
    for key, default in keys_and_default_values.items():
        values.append(remaining.pop(key, default))
    return (*values, remaining)


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