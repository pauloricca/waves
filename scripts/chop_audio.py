#!/usr/bin/env python3
"""
Audio Chopper Script

Chops a WAV file into segments based on silence detection.
Saves each non-silent segment as a separate file.

Usage:
    ./chop_audio.py <input_wav_file> <output_directory> [--threshold THRESHOLD] [--min-silence-duration MIN_DUR]

Arguments:
    input_wav_file: Path to the input WAV file
    output_directory: Directory where chopped files will be saved (created if it doesn't exist)
    --threshold: Amplitude threshold for silence detection (0.0 to 1.0, default: 0.01)
    --min-silence-duration: Minimum silence duration in seconds to consider a split (default: 0.1)

Example:
    ./chop_audio.py samples/recording.wav output/chopped
    ./chop_audio.py samples/recording.wav output/chopped --threshold 0.02 --min-silence-duration 0.2
"""

import sys
import os
import argparse
import numpy as np
from scipy.io import wavfile


def detect_silence_boundaries(audio_data, sample_rate, threshold=0.01, min_silence_duration=0.1):
    """
    Detect silence boundaries in audio data.
    
    Args:
        audio_data: NumPy array of audio samples (normalized to [-1, 1])
        sample_rate: Sample rate of the audio
        threshold: Amplitude threshold below which audio is considered silence
        min_silence_duration: Minimum duration of silence in seconds to consider a split
        
    Returns:
        List of tuples (start_sample, end_sample) for non-silent segments
    """
    # Calculate absolute amplitude
    abs_audio = np.abs(audio_data)
    
    # Determine which samples are silent
    is_silent = abs_audio < threshold
    
    # Find transitions between silence and non-silence
    # Pad with True at both ends to handle edge cases
    padded = np.pad(is_silent, (1, 1), constant_values=True)
    diff = np.diff(padded.astype(int))
    
    # diff == -1 means transition from silence to sound (start of segment)
    # diff == 1 means transition from sound to silence (end of segment)
    segment_starts = np.where(diff == -1)[0]
    segment_ends = np.where(diff == 1)[0]
    
    # Filter out very short segments (likely noise)
    min_silence_samples = int(min_silence_duration * sample_rate)
    
    segments = []
    for start, end in zip(segment_starts, segment_ends):
        # Check if this segment is substantial enough
        if end - start > min_silence_samples:
            segments.append((start, end))
    
    return segments


def load_wav_file(filepath):
    """
    Load a WAV file and normalize to float32 in range [-1, 1].
    
    Returns:
        Tuple of (sample_rate, audio_data)
    """
    sample_rate, data = wavfile.read(filepath)
    
    # Convert stereo to mono if necessary
    if data.ndim > 1:
        data = data[:, 0]
    
    # Normalize to [-1, 1]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32767.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483647.0
    elif data.dtype == np.float32:
        # Already float, but ensure it's in correct range
        data = np.clip(data, -1.0, 1.0)
    else:
        # Try to normalize based on dtype max
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    
    return sample_rate, data


def save_wav_file(filepath, sample_rate, audio_data):
    """
    Save audio data as a WAV file in 16-bit PCM format.
    """
    # Ensure audio is in correct range
    audio_data = np.clip(audio_data, -1.0, 1.0)
    
    # Convert to 16-bit PCM
    audio_int16 = np.int16(audio_data * np.iinfo(np.int16).max)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Write file
    wavfile.write(filepath, sample_rate, audio_int16)


def chop_audio_file(input_file, output_dir, threshold=0.00001, min_silence_duration=0.05):
    """
    Chop an audio file based on silence detection.
    
    Args:
        input_file: Path to input WAV file
        output_dir: Directory to save chopped files
        threshold: Amplitude threshold for silence detection
        min_silence_duration: Minimum silence duration in seconds
    """
    # Load the audio file
    print(f"Loading {input_file}...")
    sample_rate, audio_data = load_wav_file(input_file)
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Duration: {len(audio_data) / sample_rate:.2f} seconds")
    print(f"  Samples: {len(audio_data)}")
    
    # Detect silence boundaries
    print(f"\nDetecting silence (threshold={threshold}, min_duration={min_silence_duration}s)...")
    segments = detect_silence_boundaries(audio_data, sample_rate, threshold, min_silence_duration)
    
    if not segments:
        print("No non-silent segments found!")
        return
    
    print(f"Found {len(segments)} segment(s)")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract base filename without extension
    base_filename = os.path.splitext(os.path.basename(input_file))[0]
    
    # Save each segment
    for i, (start, end) in enumerate(segments, start=1):
        segment_data = audio_data[start:end]
        duration = len(segment_data) / sample_rate
        
        # Generate output filename
        output_filename = f"{base_filename}_chop_{i:03d}.wav"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save the segment
        save_wav_file(output_path, sample_rate, segment_data)
        
        print(f"  Saved {output_filename} (duration: {duration:.2f}s, samples: {start}-{end})")
    
    print(f"\nDone! Saved {len(segments)} file(s) to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Chop a WAV file into segments based on silence detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s samples/recording.wav output/chopped
  %(prog)s samples/recording.wav output/chopped --threshold 0.02
  %(prog)s samples/recording.wav output/chopped --threshold 0.02 --min-silence-duration 0.2
        """
    )
    
    parser.add_argument(
        "input_file",
        help="Path to the input WAV file"
    )
    
    parser.add_argument(
        "output_dir",
        help="Directory where chopped files will be saved (created if it doesn't exist)"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.00001,
        help="Amplitude threshold for silence detection (0.0 to 1.0, default: 0.00001)"
    )
    
    parser.add_argument(
        "--min-silence-duration",
        type=float,
        default=0.05,
        help="Minimum silence duration in seconds to consider a split (default: 0.05)"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found!")
        sys.exit(1)
    
    # Validate input file is a WAV file
    if not args.input_file.lower().endswith('.wav'):
        print(f"Error: Input file must be a WAV file!")
        sys.exit(1)
    
    # Validate threshold
    if not 0.0 <= args.threshold <= 1.0:
        print(f"Error: Threshold must be between 0.0 and 1.0!")
        sys.exit(1)
    
    # Validate min silence duration
    if args.min_silence_duration < 0:
        print(f"Error: Minimum silence duration must be positive!")
        sys.exit(1)
    
    # Chop the audio
    try:
        chop_audio_file(
            args.input_file,
            args.output_dir,
            args.threshold,
            args.min_silence_duration
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
