import os
import shutil
import math
import sounddevice as sd
import numpy as np
from scipy.io import wavfile


from config import *
from nodes.node_utils.base_node import BaseNodeModel

def play(wave):
    wave = np.clip(wave, -1, 1)
    sd.play(wave, samplerate=SAMPLE_RATE, blocking=True)


def save(wave, filename):
    """
    Save audio wave to WAV file. Supports both mono and stereo.
    
    Args:
        wave: 1D array for mono, or 2D array of shape (num_samples, num_channels) for stereo/multichannel
        filename: Output filename
    """
    # Clip wave to valid range
    wave = np.clip(wave, -1, 1)
    wave = wave.astype(np.float32)
    
    # Convert to 16-bit PCM
    wave_int16 = np.int16(wave * np.iinfo(np.int16).max)
    
    output_path = os.path.join(os.path.dirname(__file__), OUTPUT_DIR, filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    wavfile.write(output_path, SAMPLE_RATE, wave_int16)
    
    # Print info about what was saved
    if wave.ndim == 1:
        print(f"Saved {filename} (mono)")
    else:
        num_channels = wave.shape[1]
        print(f"Saved {filename} ({num_channels} channels)")



# Cache for loaded WAV files
_wav_file_cache = {}

def load_wav_file(filename):
    # Check if file is already in cache
    if filename in _wav_file_cache:
        return _wav_file_cache[filename]
    
    filepath = os.path.join(os.path.dirname(__file__), filename)
    sample_rate, data = wavfile.read(filepath)
    if data.ndim > 1:
        # If stereo, take only one channel
        data = data[:, 0]
    data = data.astype(np.float32) / 32767.0  # Normalize to [-1, 1]
    
    # Store in cache before returning
    _wav_file_cache[filename] = data
    return data


has_printed_visualisation = False
def visualise_wave(wave, do_normalise = False, replace_previous = False, extra_lines = 0):
    global has_printed_visualisation

    if DO_ONLY_VISUALISE_ONE_BUFFER:
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

    # Build entire visualization as a single string buffer to minimize print calls
    output_buffer = ""

    if replace_previous and has_printed_visualisation:
        # Move cursor up and clear lines - these need to be done without newlines
        for _ in range(full_visualisation_height + extra_lines):
            output_buffer += "\033[1A\x1b[2K"

    # Create a histogram-like visualization (floor the height to make sure we have an even number of rows)
    visualization_lines = []
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
                    line += "▁" if i == (visualisation_height_resolution_halved / 4) - 1 else  " " 
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
        visualization_lines.append(line)
    
    # Append the visualization lines with newlines
    output_buffer += '\n'.join(visualization_lines)

    # Single print call with all content - efficient and preserves cursor positioning
    print(output_buffer, flush=True)

    has_printed_visualisation = True


def look_for_duration(model: BaseNodeModel):
    """
    Recursively looks for the duration attribute in the model or its attributes.
    Returns None if no finite duration is found (e.g., for infinite-running nodes).
    If multiple durations are found (e.g., in lists), returns the largest value.
    """
    durations = []
    
    # Check if model itself has a duration attribute
    if hasattr(model, "duration") and model.duration is not None:
        # If duration is infinite, treat it as None (no duration limit)
        if not math.isinf(model.duration):
            durations.append(model.duration)
    
    # Check all attributes
    for attr in model.__dict__.values():
        if isinstance(attr, BaseNodeModel):
            # For BaseNodeModel attributes, recurse
            duration = look_for_duration(attr)
            if duration is not None:
                durations.append(duration)
        elif isinstance(attr, list):
            # For list attributes, check each item
            for item in attr:
                if isinstance(item, BaseNodeModel):
                    duration = look_for_duration(item)
                    if duration is not None:
                        durations.append(duration)
    
    # Also check __pydantic_extra__ for models with extra='allow' (like ExpressionNodeModel)
    if hasattr(model, '__pydantic_extra__') and model.__pydantic_extra__:
        for attr in model.__pydantic_extra__.values():
            if isinstance(attr, BaseNodeModel):
                duration = look_for_duration(attr)
                if duration is not None:
                    durations.append(duration)
            elif isinstance(attr, list):
                for item in attr:
                    if isinstance(item, BaseNodeModel):
                        duration = look_for_duration(item)
                        if duration is not None:
                            durations.append(duration)
    
    # Return the largest duration found, or None if no durations found
    return max(durations) if durations else None


def add_waves(a: np.ndarray, b: np.ndarray, b_offset: int = 0) -> np.ndarray:
    """
    Adds two waves together, offsetting the second wave by b_offset, increasing the length of the first wave if necessary.
    """
    if b_offset < 0:
        # For negative offsets, we need to pad the beginning of 'a' and adjust the offset for 'b'
        a = np.pad(a, (-b_offset, 0), mode='constant')
    elif b_offset > 0:
        # For positive offsets, we pad the beginning of 'b' with zeros
        b = np.pad(b, (b_offset, 0), mode='constant')
    
    if len(a) < len(b):
        # If 'a' is shorter than 'b', we need to pad 'a' at the end
        a = np.pad(a, (0, len(b) - len(a)), mode='constant')
    elif len(a) > len(b):
        # If 'a' is longer than 'b', we need to pad 'b' at the end
        b = np.pad(b, (0, len(a) - len(b)), mode='constant')
    
    # Now we can safely add the two arrays
    return np.add(a, b, out=a, casting='unsafe')


def multiply_waves(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Multiplies two waves together, offsetting the second wave by b_offset, increasing the length of the first wave if necessary.
    """
    if len(a) < len(b):
        # If 'a' is shorter than 'b', we need to pad 'a' at the end
        a = np.pad(a, (0, len(b) - len(a)), mode='constant', constant_values=1)
    elif len(a) > len(b):
        # If 'a' is longer than 'b', we need to pad 'b' at the end
        b = np.pad(b, (0, len(a) - len(b)), mode='constant', constant_values=1)
    
    # Now we can safely multiply the two arrays
    return np.multiply(a, b, out=a, casting='unsafe')