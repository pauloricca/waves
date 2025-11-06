import os
import shutil
import math
import sounddevice as sd
import numpy as np
from scipy.io import wavfile


from config import *


def detect_triggers(trigger_wave: np.ndarray, last_value: float, threshold: float = 0.5) -> tuple[list[int], float]:
    """
    Detect 0→1 crossings (triggers) in a wave signal.
    
    Args:
        trigger_wave: NumPy array to analyze for triggers
        last_value: The last value from the previous chunk (for edge detection across chunks)
        threshold: Threshold value for detecting crossings (default: 0.5)
    
    Returns:
        A tuple of (trigger_indices, new_last_value) where:
        - trigger_indices: List of sample indices where triggers occurred in this chunk
        - new_last_value: The last value in this chunk (to pass to next call)
    
    A trigger is detected when the value crosses the threshold from below to above.
    """
    triggers = []
    
    if len(trigger_wave) == 0:
        return triggers, last_value
    
    # Check first sample against last value from previous chunk
    if last_value < threshold and trigger_wave[0] >= threshold:
        triggers.append(0)
    
    # Check for crossings within the chunk
    for i in range(1, len(trigger_wave)):
        if trigger_wave[i-1] < threshold and trigger_wave[i] >= threshold:
            triggers.append(i)
    
    # Return the last value for next chunk
    new_last_value = trigger_wave[-1]
    
    return triggers, new_last_value


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
    if not is_stereo(wave):
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


def look_for_duration(model):
    """
    Recursively looks for the duration attribute in the model or its attributes.
    Returns None if no finite duration is found (e.g., for infinite-running nodes).
    If multiple durations are found (e.g., in lists), returns the largest value.
    """
    # Import here to avoid circular dependency
    from nodes.node_utils.base_node import BaseNodeModel
    
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


def match_length(array: np.ndarray, target_length: int) -> np.ndarray:
    """
    Ensures array matches target_length by padding or cropping.
    If array is shorter, pads with edge values (repeats last value).
    If array is longer, crops to target_length.
    
    Args:
        array: Input array to match length
        target_length: Desired length
        
    Returns:
        Array of length target_length
    """
    if len(array) < target_length:
        # Pad with last value
        return np.pad(array, (0, target_length - len(array)), mode='edge')
    elif len(array) > target_length:
        # Crop to target length
        return array[:target_length]
    return array


def empty_mono() -> np.ndarray:
    """
    Returns an empty mono audio array.
    Used to signal end of rendering or empty output.
    
    Returns:
        Empty 1D float32 array
    """
    return np.array([], dtype=np.float32)


def empty_stereo() -> np.ndarray:
    """
    Returns an empty stereo audio array.
    Used to signal end of rendering or empty output for stereo nodes.
    
    Returns:
        Empty 2D float32 array with shape (0, 2)
    """
    return np.array([], dtype=np.float32).reshape(0, 2)


def time_to_samples(seconds: float) -> int:
    """
    Converts time in seconds to number of samples.
    
    Args:
        seconds: Time duration in seconds
        
    Returns:
        Number of samples (rounded down to nearest integer)
    """
    return int(seconds * SAMPLE_RATE)


def samples_to_time(samples: int) -> float:
    """
    Converts number of samples to time in seconds.
    
    Args:
        samples: Number of samples
        
    Returns:
        Time duration in seconds
    """
    return samples / SAMPLE_RATE


def ensure_array(value, num_samples: int) -> np.ndarray:
    """
    Ensures a value is a numpy array of the specified length.
    If value is already an array, returns it as-is.
    If value is a scalar, creates an array filled with that value.
    
    Args:
        value: Scalar value or numpy array
        num_samples: Desired length for scalar conversion
        
    Returns:
        Numpy array of length num_samples
    """
    if isinstance(value, np.ndarray):
        return value
    return np.full(num_samples, value, dtype=np.float32)


def get_last_or_default(array: np.ndarray, default=0):
    """
    Safely gets the last value from an array, or returns default if empty.
    
    Args:
        array: Numpy array to get last value from
        default: Default value to return if array is empty
        
    Returns:
        Last value of array, or default if empty
    """
    return array[-1] if len(array) > 0 else default


def is_stereo(wave: np.ndarray) -> bool:
    """
    Check if a wave is in stereo format (2D array).
    
    Args:
        wave: Audio wave array to check
        
    Returns:
        True if stereo (2D), False if mono (1D)
    """
    return wave.ndim == 2


def to_stereo(wave: np.ndarray) -> np.ndarray:
    """
    Converts a mono or stereo wave to stereo format.
    If already stereo, returns as-is.
    If mono, duplicates the signal to both channels (center panned).
    
    Args:
        wave: 1D mono array or 2D stereo array
        
    Returns:
        2D stereo array of shape (num_samples, 2)
    """
    if is_stereo(wave):
        # Already stereo, return as-is
        return wave
    else:
        # Mono to stereo - duplicate channels (center panned)
        return np.stack([wave, wave], axis=-1)


def to_mono(wave: np.ndarray) -> np.ndarray:
    """
    Converts a mono or stereo wave to mono format.
    If already mono, returns as-is.
    If stereo, mixes both channels equally (averaging).
    
    Args:
        wave: 1D mono array or 2D stereo array
        
    Returns:
        1D mono array
    """
    if not is_stereo(wave):
        # Already mono, return as-is
        return wave
    else:
        # Stereo to mono - average both channels
        return np.mean(wave, axis=1).astype(np.float32)