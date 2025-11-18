"""
Display statistics during playback.
Handles CPU usage, elapsed time, and recording status display.
"""
import os
from random import random
import time
import numpy as np
from collections import deque

from config import *
from utils import get_cached_terminal_size, visualise_wave
from nodes.node_utils.midi_utils import get_last_midi_message_display
from nodes.node_utils.monitor_registry import create_monitor_meter


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
        print(f"Average CPU usage: {avg_cpu:.2f}%")





def format_stats_line(cpu_usage_percent: float, elapsed_seconds: float, is_recording: bool, loudness: float = 0.0, loudness_left: float = None, loudness_right: float = None, midi_message: str = None) -> str:
    """
    Format the statistics line with CPU usage, elapsed time, and recording status.
    
    Args:
        cpu_usage_percent: CPU usage as a percentage (0-100+)
        elapsed_seconds: Elapsed time in seconds since start
        is_recording: Whether recording is currently active
        loudness: Current loudness level (0.0 to 1.0+) - used for mono or max of stereo
        loudness_left: Optional left channel loudness for stereo display
        loudness_right: Optional right channel loudness for stereo display
        midi_message: Optional MIDI message display string
    
    Returns:
        Formatted string for display
    """
    # Build parts list starting with optional loudness meter
    parts = []
    
    # Add loudness meter (stereo or mono)
    if loudness_left is not None and loudness_right is not None:
        # Stereo meter: "L " (2) + meter (10) + " R " (3) + meter (9) = 24 chars
        meter_l = create_monitor_meter(loudness_left, 0.0, 1.0, width=10, color_scheme="level")
        meter_r = create_monitor_meter(loudness_right, 0.0, 1.0, width=9, color_scheme="level")
        loudness_meter = f"L {meter_l} R {meter_r}"
    else:
        # Mono meter
        loudness_meter = create_monitor_meter(loudness, 0.0, 1.0, width=24, color_scheme="level")
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
                # Note: visualised_wave_buffer contains stereo data if available (from waves.py)
                buffer_array = np.array(visualised_wave_buffer)
                
                # Check if buffer contains stereo data (2D array with 2 columns)
                if buffer_array.ndim == 2 and buffer_array.shape[1] == 2:
                    # Stereo: calculate separate L/R loudness
                    loudness_left = np.max(np.abs(buffer_array[:, 0])) if len(buffer_array) > 0 else 0.0
                    loudness_right = np.max(np.abs(buffer_array[:, 1])) if len(buffer_array) > 0 else 0.0
                    loudness = max(loudness_left, loudness_right)
                else:
                    # Mono
                    loudness = np.max(np.abs(buffer_array)) if len(buffer_array) > 0 else 0.0
                    loudness_left = None
                    loudness_right = None
                
                # Get last MIDI message for display
                midi_message = get_last_midi_message_display()
                
                # Format stats line - show loudness meter only when NOT visualizing
                stats_text = format_stats_line(
                    cpu_usage_percent,
                    elapsed,
                    recording_active_ref[0],
                    loudness,
                    loudness_left,
                    loudness_right,
                    midi_message=midi_message
                )
                
                # Build all output into a buffer first
                output_buffer = []
                current_lines = 0
                
                # Get terminal width for padding
                term_width = get_cached_terminal_size().columns
                
                # Add visualization if enabled
                if DO_VISUALISE_OUTPUT:
                    # Convert buffer to numpy array and to mono for waveform display
                    buffer_for_viz = np.array(visualised_wave_buffer)
                    # Import to_mono from utils
                    from utils import to_mono as convert_to_mono
                    mono_buffer = convert_to_mono(buffer_for_viz)
                    
                    # Render visualization to buffer (returns a string with newlines)
                    viz_output = visualise_wave(
                        mono_buffer,
                        do_normalise=False
                    )
                    
                    # Count visualization lines
                    viz_lines = viz_output.count('\n') + 1 if viz_output else 0
                    
                    # Get stats/monitor lines
                    stats_monitor_lines = []
                    if DISPLAY_RENDER_STATS:
                        from nodes.node_utils.monitor_registry import get_monitor_registry
                        monitor_lines = get_monitor_registry().get_display_lines()
                        
                        if monitor_lines:
                            stats_monitor_lines.extend(monitor_lines)
                            stats_monitor_lines.append("─" * 80)
                        stats_monitor_lines.append(stats_text)
                    
                    # Calculate total lines that will be printed
                    total_lines = viz_lines + len(stats_monitor_lines)

                    # Randomly reduce number of lines displayed occasionally as a glitch effect
                    if random() < CHANCE_OF_CLEARING_ONE_LESS_ROW:
                        last_printed_lines = max(0, last_printed_lines - 1)
                    
                    # Clear previous output if needed
                    if last_printed_lines > 0:
                        clear_codes = "".join("\033[1A\x1b[2K" for _ in range(last_printed_lines))
                        output_buffer.append(clear_codes)
                    
                    # Add visualization
                    output_buffer.append(viz_output)
                    
                    # Add stats/monitor lines
                    if DISPLAY_RENDER_STATS and stats_monitor_lines:
                        output_buffer.append("\n")  # One newline to separate from viz
                        output_buffer.append("\n".join(line.ljust(term_width) for line in stats_monitor_lines))
                    
                    current_lines = total_lines
                
                elif DISPLAY_RENDER_STATS:
                    # Build stats-only output
                    from nodes.node_utils.monitor_registry import get_monitor_registry
                    monitor_lines = get_monitor_registry().get_display_lines()
                    
                    stats_monitor_lines = []
                    if monitor_lines:
                        stats_monitor_lines.extend(monitor_lines)
                        stats_monitor_lines.append("─" * 80)
                    stats_monitor_lines.append(stats_text)
                    
                    total_lines = len(stats_monitor_lines)
                    
                    # Clear previous output if needed
                    if last_printed_lines > 0:
                        clear_codes = "".join("\033[1A\x1b[2K" for _ in range(last_printed_lines))
                        output_buffer.append(clear_codes)
                    
                    # Add stats/monitor lines
                    output_buffer.append("\n".join(line.ljust(term_width) for line in stats_monitor_lines))
                    
                    current_lines = total_lines
                
                # Print all buffered output at once
                if output_buffer:
                    print("".join(output_buffer), flush=True)
                
                # Update last_printed_lines for next iteration
                last_printed_lines = current_lines
            except Exception:
                # Silently ignore visualization errors to avoid breaking audio
                pass
        
        time.sleep(1.0 / VISUALISATION_FPS)  # 20 FPS for visualization
