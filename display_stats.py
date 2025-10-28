"""
Display statistics and visualization for real-time audio playback.
Handles CPU usage, elapsed time, and recording status display.
"""
import time
import numpy as np
from collections import deque
import os
import shutil

from config import BUFFER_SIZE, SAMPLE_RATE, DO_VISUALISE_OUTPUT, DISPLAY_RENDER_TIME_PERCENTAGE
from utils import visualise_wave
from nodes.node_utils.midi_utils import get_last_midi_message_display


def create_loudness_meter(loudness: float, width: int = 20) -> str:
    """
    Create a visual loudness meter using Unicode block characters.
    
    Args:
        loudness: Loudness level from 0.0 to 1.0 (and potentially higher for clipping)
        width: Width of the meter in characters
    
    Returns:
        A string representing the loudness meter with colors
    """
    # Clamp loudness to max 1.0 for display to keep meter width fixed
    display_loudness = min(loudness, 1.0)
    
    # Calculate how many blocks to fill
    filled = display_loudness * width
    full_blocks = int(filled)
    partial = filled - full_blocks
    
    # Choose partial block character based on fractional part
    # Unicode block elements: ▏▎▍▌▋▊▉█
    partial_chars = ['', '▏', '▎', '▍', '▌', '▋', '▊', '▉']
    partial_idx = int(partial * len(partial_chars))
    partial_char = partial_chars[partial_idx] if partial_idx < len(partial_chars) else ''
    
    # Build the meter
    # Use different colors based on level:
    # Green (0-0.7), Yellow (0.7-0.9), Red (0.9+)
    if loudness < 0.7:
        color_code = '\033[92m'  # Green
    elif loudness < 0.9:
        color_code = '\033[93m'  # Yellow
    else:
        color_code = '\033[91m'  # Red
    
    reset_code = '\033[0m'
    
    # Full blocks
    filled_part = '█' * full_blocks + partial_char
    # Empty part
    empty_part = ' ' * (width - full_blocks - (1 if partial_char else 0))
    
    return f"[{color_code}{filled_part}{reset_code}{empty_part}]"


def format_stats_line(cpu_usage_percent: float, elapsed_seconds: float, is_recording: bool, loudness: float = 0.0, show_loudness: bool = False, midi_message: str = None) -> str:
    """
    Format the statistics line with CPU usage, elapsed time, and recording status.
    
    Args:
        cpu_usage_percent: CPU usage as a percentage (0-100+)
        elapsed_seconds: Elapsed time in seconds since start
        is_recording: Whether recording is currently active
        loudness: Current loudness level (0.0 to 1.0+)
        show_loudness: Whether to show the loudness meter
        midi_message: Optional MIDI message display string
    
    Returns:
        Formatted string for display
    """
    # Build parts list starting with optional loudness meter
    parts = []
    
    # Add loudness meter if requested
    if show_loudness:
        loudness_meter = create_loudness_meter(loudness, width=20)
        parts.append(loudness_meter)
    
    # Format CPU usage
    cpu_text = f"CPU usage: {cpu_usage_percent:.2f}%"
    parts.append(cpu_text)
    
    # Format elapsed time as min:sec
    elapsed_minutes = int(elapsed_seconds // 60)
    elapsed_secs = int(elapsed_seconds % 60)
    elapsed_text = f"{elapsed_minutes}:{elapsed_secs:02d}"
    parts.append(elapsed_text)
    
    # Add MIDI message if available
    if midi_message:
        parts.append(midi_message)
    
    # Build recording indicator with slow blink (twice per second)
    if is_recording:
        # Blink at 0.5 Hz (twice per second) - on for 0.5s, off for 0.5s
        blink_on = (int(elapsed_seconds * 2) % 2) == 0
        # Red circle ANSI codes: \033[91m for red, \033[0m to reset
        # Use ● (U+25CF) for filled circle
        red_dot = "\033[91m●\033[0m" if blink_on else " "
        recording_text = f"{red_dot} recording"
        parts.append(recording_text)
    
    # Combine with vertical bars and spacing
    return "  |  ".join(parts)


def run_visualizer_and_stats(
    visualised_wave_buffer: deque,
    should_stop_flag,
    start_time: float,
    last_render_time_ref,
    recording_active_ref
):
    """
    Run visualization and stats display loop.
    
    Args:
        visualised_wave_buffer: Deque containing recent audio samples for visualization
        should_stop_flag: Reference to boolean flag indicating when to stop
        start_time: Start time in seconds (from time.time())
        last_render_time_ref: Reference to last render time value
        recording_active_ref: Reference to recording active status
    """
    
    # Lower priority for visualization thread to avoid interfering with audio
    try:
        os.nice(10)  # Increase niceness (lower priority) on Unix systems
    except (AttributeError, OSError):
        pass  # Windows or permission issues
    
    while not should_stop_flag[0]:
        if len(visualised_wave_buffer) > 0:
            try:
                # Calculate elapsed time
                elapsed = time.time() - start_time
                
                # Calculate CPU usage percentage
                cpu_usage_percent = 100 * last_render_time_ref[0] / (BUFFER_SIZE / SAMPLE_RATE)
                
                # Calculate loudness (peak of recent samples)
                buffer_array = np.array(visualised_wave_buffer)
                loudness = np.max(np.abs(buffer_array)) if len(buffer_array) > 0 else 0.0
                
                # Get last MIDI message for display
                midi_message = get_last_midi_message_display()
                
                # Format stats line - show loudness meter only when NOT visualizing
                stats_text = format_stats_line(
                    cpu_usage_percent,
                    elapsed,
                    recording_active_ref[0],
                    loudness,
                    show_loudness=not DO_VISUALISE_OUTPUT,
                    midi_message=midi_message
                )
                
                if DO_VISUALISE_OUTPUT:
                    visualise_wave(
                        np.array(visualised_wave_buffer),
                        do_normalise=False,
                        replace_previous=True,
                        extra_lines=1
                    )
                    if DISPLAY_RENDER_TIME_PERCENTAGE:
                        print(stats_text, flush=True)
                elif DISPLAY_RENDER_TIME_PERCENTAGE:
                    # Clear line and print stats only (no visualization)
                    # Fill the rest of the terminal width with spaces to blank it out
                    try:
                        term_width = shutil.get_terminal_size((80, 20)).columns
                    except Exception:
                        term_width = 80
                    padded_text = stats_text.ljust(term_width)
                    print(f"\r{padded_text}", end='', flush=True)
            except Exception:
                # Silently ignore visualization errors to avoid breaking audio
                pass
        
        time.sleep(1.0 / 20)  # 20 FPS for visualization
