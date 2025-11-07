"""
Display statistics during playback.
Handles CPU usage, elapsed time, and recording status display.
"""
import sys
import os
import time
import shutil
import numpy as np
from collections import deque

from config import *
from utils import visualise_wave
from nodes.node_utils.midi_utils import get_last_midi_message_display


# Global rolling average tracker for CPU usage
cpu_usage_samples = deque(maxlen=1000)  # Keep last 1000 samples for rolling average


def get_average_cpu_usage() -> float:
    """Calculate the average CPU usage from collected samples."""
    if len(cpu_usage_samples) == 0:
        return 0.0
    return sum(cpu_usage_samples) / len(cpu_usage_samples)


def clear_cpu_usage_samples():
    """Clear CPU usage samples (useful for hot reload or new playback)."""
    cpu_usage_samples.clear()


def print_average_cpu_usage():
    """Print the average CPU usage. Called on exit."""
    avg_cpu = get_average_cpu_usage()
    if len(cpu_usage_samples) > 0:
        print(f"\n{'='*60}")
        print(f"Average CPU usage: {avg_cpu:.2f}% (based on {len(cpu_usage_samples)} samples)")
        print('='*60)


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
    # Use grey color for empty blocks
    grey_code = '\033[90m'
    empty_part = f"{grey_code}{'|' * (width - full_blocks - (1 if partial_char else 0))}{reset_code}"
    
    return f"{color_code}{filled_part}{reset_code}{empty_part}"


def format_stats_line(cpu_usage_percent: float, elapsed_seconds: float, is_recording: bool, loudness: float = 0.0, midi_message: str = None) -> str:
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
    
    # Add loudness meter
    loudness_meter = create_loudness_meter(loudness, width=24)
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
    
    # Track how many lines we printed last time (for proper clearing)
    last_printed_lines = 0
    
    while not should_stop_flag[0]:
        if len(visualised_wave_buffer) > 0:
            try:
                # Calculate elapsed time
                elapsed = time.time() - start_time
                
                # Calculate CPU usage percentage
                cpu_usage_percent = 100 * last_render_time_ref[0] / (BUFFER_SIZE / SAMPLE_RATE)
                
                # Track CPU usage for rolling average
                cpu_usage_samples.append(cpu_usage_percent)
                
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
                    midi_message=midi_message
                )
                
                if DO_VISUALISE_OUTPUT:
                    visualise_wave(
                        np.array(visualised_wave_buffer),
                        do_normalise=False,
                        replace_previous=True,
                        extra_lines=1
                    )
        
                if DISPLAY_RENDER_STATS:
                    # Clear line and print stats only (no visualization)
                    # Get monitored nodes
                    from nodes.node_utils.monitor_registry import get_monitor_registry
                    monitor_lines = get_monitor_registry().get_display_lines()
                    
                    # Build output with monitored nodes above main stats
                    output_lines = []
                    if monitor_lines:
                        output_lines.extend(monitor_lines)
                        # Add separator line
                        output_lines.append("─" * 80)
                    output_lines.append(stats_text)
                    
                    # Get terminal width for padding
                    try:
                        term_width = shutil.get_terminal_size((80, 20)).columns
                    except Exception:
                        term_width = 80
                    
                    # Move cursor up to overwrite previous output if we printed before
                    if last_printed_lines > 0:
                        print(f"\033[{last_printed_lines}A", end='')
                    
                    # Print all lines with padding
                    for line in output_lines:
                        padded_line = line.ljust(term_width)
                        print(padded_line, flush=True)
                    
                    # Update line count for next iteration
                    last_printed_lines = len(output_lines)
            except Exception:
                # Silently ignore visualization errors to avoid breaking audio
                pass
        
        time.sleep(1.0 / VISUALISATION_FPS)  # 20 FPS for visualization
