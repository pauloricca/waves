import numpy as np
import os
import sounddevice as sd
from scipy.io import wavfile
from scipy import signal
import yaml
import shutil
import sys
import time
from enum import Enum
from scipy.interpolate import PchipInterpolator
import traceback

sounds = {}
YAML_FILE = "waves.yaml"
OUTPUT_DIR = "output"
ENVELOPE_TYPE = "exponential"  # Options: "linear", "exponential"
SAMPLE_RATE = 44100
VISUALISATION_ROW_HEIGHT = 10
DIVIDE_BY = 2
DO_FREQ_CLOUD = False
DO_NORMALISE_EACH_SOUND = False
DO_SHUFFLE_WAVES = False

class WaveFunction(Enum):
    SINE = "sin"
    COSINE = "cos"
    SQUARE = "sqr"
    TRIANGLE = "tri"
    SAWTOOTH = "saw"
    NOISE = "noise"

class InterpolationTypes(Enum):
    LINEAR = "linear"
    SMOOTH = "smooth"
    STEP = "step"

# Interpolates a list of values or a list of lists with relative positions
def interpolate_values(values, num_samples, interpolation_type):
    # If all the values aparet from the first and last are lists, we assume they are relative positions
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

# Returns a value or a wave, for a given attribute
def get_wavable_value(object, attribute_name, duration, default_value=None):
    value = object.get(attribute_name, default_value)
    # Linear interpolation of values in a list
    if isinstance(value, list):
        interpolation_type = object.get("interpolation", InterpolationTypes.SMOOTH.value)
        num_samples = int(duration * SAMPLE_RATE)
        value = interpolate_values(value, num_samples, interpolation_type)
    # Wave object
    if isinstance(value, dict):
        value = generate_wave(value, duration=duration)
    return value

def generate_wave(sound_obj, amplitude=1, duration=None, frequency=None, do_normalize=True):
    resolve_wave_syntax_sugars(sound_obj)
    duration = sound_obj.get("duration", 1) * (duration if duration else 1)
    release = min(1, sound_obj.get("release", 0)) * duration
    attack = min(1, sound_obj.get("attack", 0)) * duration
    min_wave_value = sound_obj.get("min", -1)
    max_wave_value = sound_obj.get("max", 1)
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)

    total_wave = 0 * t

    base_freq = (
        frequency
        if frequency is not None
        else get_wavable_value(sound_obj, "freq", duration, 0)
    )

    base_amp = get_wavable_value(sound_obj, "amp", duration, 1) * amplitude

    for wave in sound_obj["waves"]:
        wave_amp = get_wavable_value(wave, "amp", duration, 1)
        wave_type = wave.get("type", WaveFunction.SINE.value)
        if "freq" in wave or wave_type == WaveFunction.NOISE.value:
            if wave_type == WaveFunction.NOISE.value:
                total_wave += base_amp * wave_amp * np.random.normal(0, 1, len(t))
            else:
                wave_freq = base_freq * get_wavable_value(wave, "freq", duration, 0)

                if wave_type in [WaveFunction.SINE.value, WaveFunction.COSINE.value]:
                    wave_function = np.sin if wave_type == WaveFunction.SINE.value else np.cos
                    # Is frequency variable?
                    if(isinstance(wave_freq, np.ndarray)):
                        dt = 1 / SAMPLE_RATE
                        # Compute cumulative phase
                        phase = 2 * np.pi * np.cumsum(wave_freq) * dt
                        total_wave += base_amp * wave_amp * wave_function(phase[:len(total_wave)])
                    else:
                        total_wave += base_amp * wave_amp * wave_function(2 * np.pi * wave_freq * t)
                elif wave_type == WaveFunction.SQUARE.value:
                    total_wave += base_amp * wave_amp * np.sign(np.sin(2 * np.pi * wave_freq * t))
                elif wave_type == WaveFunction.TRIANGLE.value:
                    total_wave += base_amp * wave_amp * (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * wave_freq * t))
                elif wave_type == WaveFunction.SAWTOOTH.value:
                    total_wave += base_amp * wave_amp * (2 / np.pi) * np.arctan(np.tan(np.pi * wave_freq * t))
                if DO_FREQ_CLOUD:
                    # Generate 15 waves with slight variations in frequency and amplitude
                    for _ in range(50):
                        freq_variation = np.random.normal(0, 0.1 * wave_freq)
                        amp_variation = np.exp(-0.5 * (freq_variation / (0.1 * wave_freq))**2)
                        varied_freq = wave_freq + freq_variation
                        varied_amp = 0.2 * base_amp * wave_amp * amp_variation
                        total_wave += varied_amp * np.sin(2 * np.pi * varied_freq * t)

        if "group" in wave:
            sub_waves = generate_wave(
                wave["group"], wave_amp, duration, wave_freq, do_normalize=False
            )

            # Pad the shorter wave to match the length of the longer one
            if len(sub_waves) > len(total_wave):
                total_wave = np.pad(total_wave, (0, len(sub_waves) - len(total_wave)))
            elif len(sub_waves) < len(total_wave):
                sub_waves = np.pad(sub_waves, (0, len(total_wave) - len(sub_waves)))

            total_wave += base_amp * sub_waves

    if release > 0:
        if ENVELOPE_TYPE == "linear":
            fade_out = np.linspace(1, 0, int(SAMPLE_RATE * release))
        else:
            fade_out = np.exp(-np.linspace(0, 5, int(SAMPLE_RATE * release)))
        total_wave[-len(fade_out) :] *= fade_out

    if attack > 0:
        if ENVELOPE_TYPE == "linear":
            fade_in = np.linspace(0, 1, int(SAMPLE_RATE * attack))
        else:
            fade_in = 1 - np.exp(-np.linspace(0, 5, int(SAMPLE_RATE * attack)))
        total_wave[: len(fade_in)] *= fade_in

    if do_normalize and DO_NORMALISE_EACH_SOUND:
        total_wave = np.clip(total_wave, -1, 1)  # Ensure wave is in the range [-1, 1]
        total_wave = total_wave.astype(np.float32)  # Convert to float32 for sounddevice

    # Convert from [-1, 1] to [min_wave_value, max_wave_value]
    total_wave = (total_wave + 1) / 2
    total_wave = total_wave * (max_wave_value - min_wave_value) + min_wave_value

    return total_wave


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
    print(f"saved {filename}")


def generate_and_save(sound_name):
    wave = generate_wave(sounds[sound_name])
    save(wave, f"{sound_name}.wav")
    return wave


# WIP, reverb gets cut off, and should create a better kernal
def add_reverb(wave):
    # Apply a convolution to create a reverb effect
    reverb_kernel = -np.exp(
        np.linspace(0, 1, int(SAMPLE_RATE * 0.2))
    )  # Exponential decay kernel
    reverb_kernel /= np.sum(reverb_kernel)  # Normalize the kernel

    combined_wave_reverb = np.convolve(wave, reverb_kernel, mode="full")
    combined_wave_reverb = combined_wave_reverb[: len(wave)]  # Trim to original length

    combined_wave_reverb *= 80

    return combined_wave_reverb + wave


def add_delay(wave, delay_time=0.1, repeats=3, feedback=0.3, do_trim=False):
    # Apply a delay effect
    delay_samples = int(SAMPLE_RATE * delay_time)
    delayed_wave = np.zeros(len(wave) + delay_samples * repeats)

    for i in range(repeats):
        delayed_wave[i * delay_samples : i * delay_samples + len(wave)] += wave * (feedback ** i)

    return delayed_wave[: len(wave)] if do_trim else delayed_wave # Trim to original length

def add_low_pass_filter(wave, cutoff_freq=1000):
    # Apply a low-pass filter to the wave
    nyquist = 0.5 * SAMPLE_RATE
    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(1, normal_cutoff, btype="low", analog=False)
    filtered_wave = signal.filtfilt(b, a, wave)
    return filtered_wave

def normalise_wave(wave):
    # Normalize the wave to the range [-1, 1]
    peak = np.max(np.abs(wave))
    if peak > 0:
        wave /= peak
    return wave


def shuffle_wave(wave, chunk_length=0.1, percentage_of_chunks_to_invert=0.2):
    # Shuffle random chunks of the wave
    chunk_size = int(SAMPLE_RATE * chunk_length)
    num_chunks = len(wave) // chunk_size

    # Split the wave into chunks
    chunks = [wave[i * chunk_size : (i + 1) * chunk_size] for i in range(num_chunks)]

    # Wave to small to shuffle, return the original wave
    if len(chunks) < 2:
        return wave

    # Shuffle the chunks
    np.random.shuffle(chunks)

    # Invert 10% of the chunks
    num_chunks_to_invert = int(percentage_of_chunks_to_invert * num_chunks)
    chunks_to_invert = np.random.choice(num_chunks, num_chunks_to_invert, replace=False)

    for idx in chunks_to_invert:
        chunks[idx] = -chunks[idx]  # Invert the chunk

    # Recombine the shuffled chunks
    combined_wave_shuffled = np.concatenate(chunks)

    return combined_wave_shuffled


def load_wav_file(filename):
    sample_rate, data = wavfile.read(filename)
    if data.ndim > 1:
        # If stereo, take only one channel
        data = data[:, 0]
    data = data.astype(np.float32) / 32767.0  # Normalize to [-1, 1]
    return data


def visualise_wave(wave):
    # Get terminal width
    visualisation_width = shutil.get_terminal_size().columns
    visualisation_height = VISUALISATION_ROW_HEIGHT * 4

    # Normalize the waveform to fit within the terminal height
    # normalized_wave = wave / np.max(np.abs(wave))
    scaled_wave = (wave * visualisation_height).astype(int)

    # Group values in scaled_wave into visualisation_width groups
    group_size = max(1, len(scaled_wave) // visualisation_width)
    grouped_wave = [
        max(abs(scaled_wave[i * group_size : (i + 1) * group_size]), default=0)
        for i in range(visualisation_width)
    ]

    # Create a histogram-like visualization
    for i in range(VISUALISATION_ROW_HEIGHT):
        line = ""
        for value in grouped_wave:
            if value >= visualisation_height - i * 4:
                line += "█"  # Full block
            elif value >= visualisation_height - i * 4 - 1:
                line += "▆"  # U+2586 Lower three-quarters block
            elif value >= visualisation_height - i * 4 - 2:
                line += "▄"  # U+2584 Lower half block
            elif value >= visualisation_height - i * 4 - 3:
                line += "▂"  # U+2582 Lower one-quarter block
            else:
                line += " "  # Empty space
        print(line)


def generate_sequence_sound(sequence_sound):
    generated_waves = {}

    sequence = sequence_sound["sequence"]
    interval = sequence_sound.get("interval", 0)
    repeat = sequence_sound.get("repeat", 1)

    # Get unique sounds (sound names with parameters) in the sequence
    unique_sounds = set()
    for sound_names in sequence:
        if isinstance(sound_names, str):
            unique_sounds.add(sound_names)
        elif isinstance(sound_names, list):
            unique_sounds.update(sound_names)

    for sound_names in unique_sounds:
        if sound_names and sound_names not in generated_waves:
            parts = sound_names.split()
            main_sound_name = parts[0]
            params = parts[1:]

            generation_params = {}
            for param in params:
                if param.startswith("f"):
                    generation_params["frequency"] = float(param[1:])
                elif param.startswith("a"):
                    generation_params["amplitude"] = float(param[1:])

            if "waves" in sounds[main_sound_name]:
                generated_waves[sound_names] = generate_wave(
                    sounds[main_sound_name], **generation_params
                )
            else:
                generated_waves[sound_names] = generate_sequence_sound(sounds[sound_names])

            save(generated_waves[sound_names], f"{sound_names}.wav")
            # play(generated_waves[sound_name])

    # Create a combined wave based on the sequence
    combined_wave = np.array([], dtype=np.float32)

    max_length = int(SAMPLE_RATE * (interval * (len(sequence) + 1))) if interval else sum([len(generated_waves[sound_names]) for sound_names in sequence])
    combined_wave = np.zeros(max_length, dtype=np.float32)

    last_end_idx = 0
    for i, sound_names in enumerate(sequence):
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

        if interval:
            start_idx = int(SAMPLE_RATE * interval * i)
        else:
            start_idx = last_end_idx

        end_idx = start_idx + len(wave)

        if end_idx > len(combined_wave):
            combined_wave = np.pad(combined_wave, (0, end_idx - len(combined_wave)))

        combined_wave[start_idx:end_idx] += wave
        last_end_idx = end_idx
    
    # Repeat the combined wave "repeat" times with "interval" seconds in between
    repeated_wave = np.array([], dtype=np.float32)
    for _ in range(repeat):
        repeated_wave = np.concatenate(
            (repeated_wave, combined_wave, np.zeros(int(SAMPLE_RATE * interval)))
        )
    combined_wave = repeated_wave

    return combined_wave


def resolve_wave_syntax_sugars(sound_obj):
    """
    A wave should have a waves list, but for commodity, if we only have a single wave, 
    we allow the wave to be defined in the root object and this function moves it to a waves list.
    E.g.:
    {
        "freq": 440,
        "amp": 1
    }
    becomes:
    {
        "waves": [
            {
                "freq": 440,
                "amp": 1
            }
        ]
    }
    """
    if "waves" not in sound_obj and "freq" in sound_obj:
        sound_obj["waves"] = [{**sound_obj}]
        sound_obj["freq"] = 1


def main():
    global sounds

    with open(YAML_FILE, "r") as file:
        sounds = yaml.safe_load(file)

    if len(sys.argv) < 2:
        print("Usage: python waves.py <sound-name>")
        sys.exit(1)
    
    sound_to_play = sys.argv[1]

    if "sequence" in sounds[sound_to_play]:
        combined_wave = generate_sequence_sound(sounds[sound_to_play])
    else:
        combined_wave = generate_wave(sounds[sound_to_play])

    # Normalize the combined wave
    peak = np.max(np.abs(combined_wave))
    combined_wave /= DIVIDE_BY

    save(combined_wave, f"{sound_to_play}.wav")

    if DO_SHUFFLE_WAVES:
        combined_wave_shuffled = shuffle_wave(combined_wave, 0.1)

        save(combined_wave_shuffled, "combined-shuffled.wav")

        # combined_wave_reverb = add_convolution(combined_wave_shuffled)

        # normalise_wave(combined_wave_reverb)

        # save(combined_wave_reverb, "combined-shuffled-reverb.wav")

        # combined_wave_super_shuffled = shuffle_wave(combined_wave_reverb, 0.3)

        # save(combined_wave_super_shuffled, "combined-shuffled-super-shuffled.wav")

        combined_wave_re_shuffled = shuffle_wave(combined_wave_shuffled, 0.3)

        save(combined_wave_re_shuffled, "combined-shuffled-re-shuffled.wav")


    print ("Peak:", peak)

    wave_to_play = combined_wave

    # wave_to_play = add_delay(wave_to_play, 0.3, 3)
    # wave_to_play = add_low_pass_filter(wave_to_play)

    wave_to_play = normalise_wave(wave_to_play)
    play(wave_to_play)


def get_file_modified_time(filepath):
    return os.path.getmtime(filepath)


if __name__ == "__main__":
    last_modified_time = ""
    while True:
        current_modified_time = get_file_modified_time(YAML_FILE)
        if current_modified_time != last_modified_time:
            last_modified_time = current_modified_time
            try:
                main()
            except Exception as e:
                print(f"Error: {e}")
                traceback.print_exc()
        time.sleep(0.5)



