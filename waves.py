import numpy as np
import os
import sounddevice as sd
from scipy.io.wavfile import write
import yaml
import shutil

sounds = {}
YAML_FILE = "waves.yaml"
OUTPUT_DIR = "output"
SAMPLE_RATE = 44100
VISUALISATION_ROW_HEIGHT = 10
DIVIDE_BY = 2
DO_FREQ_CLOUD = False
DO_NORMALISE_EACH_SOUND = True


def generate(sound_obj, amplitude=1, duration=None, frequency=None, do_normalize=True):
    duration = (sound_obj["duration"] if "duration" in sound_obj else 1) * (
        duration if duration else 1
    )
    release = (sound_obj["release"] * duration) if "release" in sound_obj else 0
    attack = (sound_obj["attack"] * duration) if "attack" in sound_obj else 0
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)

    total_wave = 0 * t

    base_freq = (
        frequency
        if frequency is not None
        else sound_obj["freq"] if "freq" in sound_obj else 0
    )
    base_amp = (sound_obj["amp"] if "amp" in sound_obj else 1) * amplitude
    for wave in sound_obj["waves"]:
        wave_amp = wave["amp"] if "amp" in wave else 1
        if "freq" in wave:
            wave_freq = base_freq * wave["freq"]
            total_wave += base_amp * wave_amp * np.sin(2 * np.pi * wave_freq * t)

            if DO_FREQ_CLOUD:
                # Generate 15 waves with slight variations in frequency and amplitude
                for _ in range(50):
                    freq_variation = np.random.normal(0, 0.1 * wave_freq)
                    amp_variation = np.exp(-0.5 * (freq_variation / (0.1 * wave_freq))**2)
                    varied_freq = wave_freq + freq_variation
                    varied_amp = 0.2 * base_amp * wave_amp * amp_variation
                    total_wave += varied_amp * np.sin(2 * np.pi * varied_freq * t)

        if "group" in wave:
            sub_waves = generate(
                wave["group"], wave_amp, duration, wave_freq, do_normalize=False
            )

            # Pad the shorter wave to match the length of the longer one
            if len(sub_waves) > len(total_wave):
                total_wave = np.pad(total_wave, (0, len(sub_waves) - len(total_wave)))
            elif len(sub_waves) < len(total_wave):
                sub_waves = np.pad(sub_waves, (0, len(total_wave) - len(sub_waves)))

            total_wave += base_amp * sub_waves

    if release > 0:
        # Apply a linear fade-out for the release phase
        fade_out = np.linspace(1, 0, int(SAMPLE_RATE * release))
        total_wave[-len(fade_out) :] *= fade_out

    if attack > 0:
        # Apply a linear fade-in for the attack phase
        fade_in = np.linspace(0, 1, int(SAMPLE_RATE * attack))
        total_wave[: len(fade_in)] *= fade_in

    if do_normalize and DO_NORMALISE_EACH_SOUND:
        total_wave = np.clip(total_wave, -1, 1)  # Ensure wave is in the range [-1, 1]
        total_wave = total_wave.astype(np.float32)  # Convert to float32 for sounddevice

    return total_wave


def play(wave):
    wave = np.clip(wave, -1, 1)
    sd.play(wave, samplerate=SAMPLE_RATE, blocking=True)


def save(wave, filename):
    # Normalize wave to 16-bit PCM format
    wave = np.clip(wave, -1, 1)  # Ensure wave is in the range [-1, 1]
    wave = wave.astype(np.float32)  # Convert to float32 for sounddevice
    wave_int16 = np.int16(wave * 32767)
    output_path = os.path.join(os.path.dirname(__file__), OUTPUT_DIR, filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write(output_path, SAMPLE_RATE, wave_int16)
    visualise_wave(wave)
    print(f"saved {filename}")


def generate_and_save(sound_name):
    wave = generate(sounds[sound_name])
    save(wave, f"{sound_name}.wav")
    return wave


def add_convolution(wave):
    # Apply a convolution to create a reverb effect
    reverb_kernel = -np.exp(
        np.linspace(0, 1, int(SAMPLE_RATE * 0.2))
    )  # Exponential decay kernel
    reverb_kernel /= np.sum(reverb_kernel)  # Normalize the kernel

    combined_wave_reverb = np.convolve(wave, reverb_kernel, mode="full")
    combined_wave_reverb = combined_wave_reverb[: len(wave)]  # Trim to original length

    combined_wave_reverb *= 80

    return combined_wave_reverb + wave


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


if __name__ == "__main__":
    with open(YAML_FILE, "r") as file:
        sounds = yaml.safe_load(file)

    sequence = [
        ["thump f80", "tick"],
        "",
        "",
        "tick",
        "",
        ["phoom", "tick"],
        "",
        "",
        "",
        "tick f18000",
        "tick",
        "phoom f15000",
        "tick f10000",
        "",
        "",
        "",
        "tick",
        "tick f19000",
        "",
        "",
        "",
        "tick",
    ]

    sequence2 = [
        ["thump f120", "tick"],
        "",
        ["thump f80"],
        "",
        "tick",
        ["phoom f18000", "tick"],
        "",
        "tick",
        "",
        "",
        "tick f19000",
        "tick",
        "",
        "beep f13000",
        "",
        "tick",
        "tick f18000",
        "",
        "tick f10000",
        "",
        "tick",
        "",
    ]

    sequence = [
        *sequence,
        *sequence2,
        *sequence,
        *sequence,
    ]

    sequence = [
        ["thump f80", "tick"],
        "",
        "tick",
        "",
        ["phoom", "tick"],
        "",
        "tick",
        "beep f13000",
        ["thump f120", "tick"],
        "tick f18000",
        "tick",
        "phoom f15000",
        "tick f10000",
        "",
        "tick",
        "",
        "",
        "tick",
        "tick",
        "",
        "",
        "",
        "tick",
        "",
        "",
        "",
        ["thump f69", "tick"],
        "",
        "tick",
        "",
        ["phoom f185", "tick"],
        "",
        "tick",
        "",
        ["thump f105", "tick"],
        "tick f18000",
        "tick",
        "phoom f15000",
        "tick f10000",
        "",
        "",
        "tick",
        "",
        "",
        "tick",
        "tick",
        "",
        "",
        "",
        "",
        "",
        "",
        ["thump f60", "tick"],
        "",
        "tick",
        "",
        ["phoom f187", "tick"],
        "beep f12000",
        "tick",
        "",
        ["thump f90", "tick"],
        "tick f18000",
        "tick",
        "phoom f15000",
        "tick f10000",
        "tick",
        "",
        "tick",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "tick",
        "",
        "",
        ["thump f70", "tick"],
        "",
        "tick",
        "",
        ["phoom", "tick"],
        "",
        "tick",
        "",
        ["thump f105", "tick"],
        ["tick f18000", "beep f11000"],
        "tick",
        "phoom f15000",
        "tick f10000",
        "",
        "tick",
        "",
        "tick",
        "",
        "",
        "tick",
        "",
        "",
        "",
        "",
        "tick",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        ["thump f120", "tick"],
        "",
        "tick",
        "",
        ["phoom", "tick"],
        "",
        "tick",
        "",
        ["thump f97.5", "tick"],
        "tick f18000",
        "tick",
        "phoom f15000",
        "tick f10000",
        "",
        "",
        "tick",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "tick",
        "",
        ["thump f105", "tick"],
        "",
        "tick",
        "",
        ["phoom", "tick"],
        "",
        "tick",
        "",
        ["thump f73", "tick"],
        "tick f18000",
        "tick",
        "phoom f15000",
        "tick f10000",
        "tick",
        "",
        "",
        "",
        "tick",
        "",
        "tick",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        ["thump f100", "tick"],
        "",
        "tick",
        "",
        ["phoom", "tick"],
        "",
        "tick",
        "",
        ["thump f60", "tick"],
        "tick f18000",
        "tick",
        "phoom f15000",
        "tick f10000",
        "tick",
        "",
        "",
        "tick",
        "",
        "",
        "tick",
        "",
        "tick",
        "",
        "",
        "",
        "",
        "",
        ["thump f90", "tick"],
        "",
        "tick",
        "",
        ["phoom", "tick"],
        "",
        "tick",
        "",
        ["thump f105", "tick"],
        "tick f18000",
        "tick",
        "phoom f15000",
        "tick f10000",
        "tick",
        "",
        "",
        "",
        "tick",
        "",
        "tick",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "tick",
        "",
        "",
        "",
        "",
    ]

    sequence_ = [
        ["thump f200", "tick"],
        "thump f220",
        "tick f280",
        "",
        ["phoom f290", "tick"],
        "",
        "tick",
        "beep f279",
        ["thump f278", "tick"],
        "thump f284",
        "tick",
        "phoom f15000",
        "thump f285",
        "",
        "tick",
        "",
        "",
        "tick"
    ]

    generated_waves = {}

    # Get unique values in the sequence
    unique_sounds = set()
    for sound_names in sequence:
        if isinstance(sound_names, str):
            unique_sounds.add(sound_names)
        else:
            unique_sounds.update(sound_names)

    for sound_names in unique_sounds:
        if sound_names != "":
            parts = sound_names.split()
            main_sound_name = parts[0]
            params = parts[1:]

            generation_params = {}
            for param in params:
                if param.startswith("f"):
                    generation_params["frequency"] = float(param[1:])
                elif param.startswith("a"):
                    generation_params["amplitude"] = float(param[1:])


            generated_waves[sound_names] = generate(
                sounds[main_sound_name], **generation_params
            )

            save(generated_waves[sound_names], f"{sound_names}.wav")
            # play(generated_waves[sound_name])

    # Create a combined wave based on the sequence
    combined_wave = np.array([], dtype=np.float32)

    max_length = int(SAMPLE_RATE * (0.1 * len(sequence) + 1))
    combined_wave = np.zeros(max_length, dtype=np.float32)

    for i, sound_names in enumerate(sequence):
        # Ensure sound_names is always a list
        if isinstance(sound_names, str):
            sound_names = [sound_names]

        sequence_of_waves = [
            generated_waves[name] if name else [] for name in sound_names
        ]
        max_wave_length = max(len(w) for w in sequence_of_waves)
        padded_waves = [
            np.pad(w, (0, max_wave_length - len(w))) for w in sequence_of_waves
        ]
        wave = sum(padded_waves)

        start_idx = int(SAMPLE_RATE * 0.1 * i)
        end_idx = start_idx + len(wave)

        if end_idx > len(combined_wave):
            combined_wave = np.pad(combined_wave, (0, end_idx - len(combined_wave)))

        combined_wave[start_idx:end_idx] += wave
    
    # Normalize the combined wave
    peak = np.max(np.abs(combined_wave))
    combined_wave /= DIVIDE_BY

    save(combined_wave, "combined.wav")

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

    while True:
        play(combined_wave)
    # play(combined_wave_shuffled)
    # play(combined_wave_reverb)
    # play(combined_wave_super_shuffled)
    # play(combined_wave_re_shuffled)


