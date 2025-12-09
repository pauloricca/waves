"""
Monitor registry for displaying real-time node output visualization.
Tracks monitored nodes and provides formatted display lines for the visualizer.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Dict
import numpy as np

if TYPE_CHECKING:
    from nodes.node_utils.base_node import BaseNode


def create_monitor_meter(value: float, min_val: float, max_val: float, width: int = 40, color_scheme: str = "level", is_bipolar: bool = False) -> str:
    """
    Create a visual meter representation of a value.
    
    Args:
        value: The current value to display
        min_val: The minimum value in the range
        max_val: The maximum value in the range
        width: The width of the meter in characters
        color_scheme: Either "level" for level-based coloring or "value" for single color
        is_bipolar: If True, display as centered meter (negative left, positive right)
    
    Returns:
        A colored string representation of the meter
    """
    # Clamp value to range
    value = max(min_val, min(max_val, value))
    
    # Bipolar mode: centered at 0
    if is_bipolar:
        # Purple color for bipolar meters
        color = '\033[95m'  # Magenta/Purple
        reset = '\033[0m'
        empty_color = '\033[90m'  # Grey
        
        # Calculate center position (ensure even width)
        half_width = width // 2
        
        # Determine if value is negative or positive
        if value > 0:
            # Positive: fill from center to right
            normalized = value / max_val if max_val > 0 else 0
            filled = normalized * half_width
            full_blocks = int(filled)
            partial = filled - full_blocks
            
            # Unicode block characters for smooth visualization
            blocks = ['▏', '▎', '▍', '▌', '▋', '▊', '▉', '█']
            partial_char = blocks[min(int(partial * 8), 7)] if partial > 0 else ''
            
            # Build: empty left + filled right + empty right
            empty_left = empty_color + ('│' * half_width) + reset
            filled_right = color + ('█' * full_blocks) + partial_char + reset
            empty_right = empty_color + ('│' * (half_width - full_blocks - (1 if partial_char else 0))) + reset
            
            return empty_left + filled_right + empty_right
        elif value < 0:
            # Negative: fill from center to left
            normalized = abs(value) / abs(min_val) if min_val < 0 else 0
            filled = normalized * half_width
            full_blocks = int(filled)
            partial = filled - full_blocks
            
            # For left-side fill, we need the partial block to appear filled from the RIGHT
            # Use inverse video to flip colors: show the complement block with inverted colors
            # e.g., to show 1/8 filled from right, show 7/8 filled from left with inverse video
            blocks = ['▏', '▎', '▍', '▌', '▋', '▊', '▉', '█']
            # Calculate complement: 1/8 becomes 7/8, 2/8 becomes 6/8, etc.
            complement_index = 7 - min(int(partial * 8), 7)
            partial_char = blocks[complement_index] if partial > 0 and complement_index < 7 else ''
            
            # Use inverse video for the partial block to flip foreground/background
            inverse = '\033[7m'  # Inverse video
            
            # Build: empty left + inverted partial (appears filled from right) + full blocks + empty right
            # The inverted partial block on the LEFT edge shows growth from left
            empty_left = empty_color + ('│' * (half_width - full_blocks - (1 if partial_char else 0))) + reset
            if partial_char:
                filled_left = inverse + color + partial_char + reset + reset + color + ('█' * full_blocks) + reset
            else:
                filled_left = color + ('█' * full_blocks) + reset
            empty_right = empty_color + ('│' * half_width) + reset
            
            return empty_left + filled_left + empty_right
        else:
            # Zero: just show empty meter
            return empty_color + ('│' * width) + reset
    
    # Standard mode: left-to-right fill
    # Normalize to 0-1 range
    normalized = (value - min_val) / (max_val - min_val) if max_val > min_val else 0
    
    # Calculate how many characters to fill
    filled = normalized * width
    full_blocks = int(filled)
    partial = filled - full_blocks
    
    # Unicode block characters for smooth visualization
    blocks = ['▏', '▎', '▍', '▌', '▋', '▊', '▉', '█']
    partial_char = blocks[min(int(partial * 8), 7)] if partial > 0 else ''
    
    # Create the meter string
    if color_scheme == "level":
        # Green-yellow-red gradient based on level
        if normalized < 0.5:
            color = '\033[92m'  # Green
        elif normalized < 0.8:
            color = '\033[93m'  # Yellow
        else:
            color = '\033[91m'  # Red
    else:
        color = '\033[94m'  # Blue
    
    reset = '\033[0m'
    empty_color = '\033[90m'  # Grey
    
    # Build the meter
    filled_part = color + ('█' * full_blocks) + partial_char + reset
    empty_part = empty_color + ('│' * (width - full_blocks - (1 if partial_char else 0))) + reset
    
    return filled_part + empty_part


class NodeMonitor:
    """
    Base monitor class that tracks a node's output and formats display line.
    Subclasses can override format_line() for custom visualization.
    """
    
    def __init__(self, node_id: str, node: BaseNode):
        self.node_id = node_id
        self.node = node
        self.peak_level = 0.0
        self.peak_left = 0.0
        self.peak_right = 0.0
        self.min_level = 0.0  # For bipolar mode
        self.min_left = 0.0  # For bipolar stereo mode
        self.min_right = 0.0  # For bipolar stereo mode
        self.is_stereo = False  # Track whether the last update was stereo
        self.is_bipolar = False  # Track whether this is a bipolar monitor
        
        # Use explicit ID if available, otherwise use the full node_id
        self.display_name = node_id
        if hasattr(node, 'model') and hasattr(node.model, 'id') and node.model.id:
            self.display_name = node.model.id
        
        # Trim display name if too long (keep right 50 chars, add "..." on left)
        if len(self.display_name) > 50:
            self.display_name = "..." + self.display_name[-50:]
    
    def update(self, output: np.ndarray):
        """Update monitor with latest output from the node"""
        if len(output) == 0:
            return
        
        # Get monitor settings from node
        use_abs = getattr(self.node, '_monitor_use_abs', True)
        self.is_bipolar = not use_abs  # Store bipolar mode for display
        
        # Check if output is stereo (2D array with 2 channels)
        if output.ndim == 2 and output.shape[1] == 2:
            self.is_stereo = True
            if use_abs:
                self.peak_left = float(np.max(np.abs(output[:, 0])))
                self.peak_right = float(np.max(np.abs(output[:, 1])))
            else:
                # For bipolar, track both positive and negative peaks
                self.peak_left = float(np.max(output[:, 0]))
                self.peak_right = float(np.max(output[:, 1]))
                self.min_left = float(np.min(output[:, 0]))
                self.min_right = float(np.min(output[:, 1]))
            self.peak_level = max(self.peak_left, self.peak_right)
        else:
            self.is_stereo = False
            if use_abs:
                self.peak_level = float(np.max(np.abs(output)))
            else:
                # For bipolar, track both positive and negative peaks
                self.peak_level = float(np.max(output))
                self.min_level = float(np.min(output))
    
    def format_line(self) -> str:
        """Format display line for this monitor. Override in subclasses for custom display."""
        # Get monitor range and color scheme from node (dynamically, so it picks up changes)
        min_val, max_val = getattr(self.node, '_monitor_range', (0.0, 1.0))
        color_scheme = getattr(self.node, '_monitor_color_scheme', 'level')
        is_bipolar = getattr(self, 'is_bipolar', False)
        
        # Total width for meters (slightly wider than main output meter)
        total_meter_width = 24
        
        if self.is_stereo:
            # Show peak values within current chunk for stereo
            meter_l = create_monitor_meter(self.peak_left, min_val, max_val, width=10, color_scheme=color_scheme, is_bipolar=is_bipolar)
            meter_r = create_monitor_meter(self.peak_right, min_val, max_val, width=9, color_scheme=color_scheme, is_bipolar=is_bipolar)
            value_str = f"L:{self.peak_left:.2f} R:{self.peak_right:.2f}"
            return f"L {meter_l} R {meter_r}  {self.display_name} ({value_str})"
        else:
            if is_bipolar:
                # For bipolar mono, show the value that's further from zero within current chunk
                display_value = self.peak_level if abs(self.peak_level) > abs(getattr(self, 'min_level', 0)) else getattr(self, 'min_level', 0)
                meter = create_monitor_meter(display_value, min_val, max_val, width=total_meter_width, color_scheme=color_scheme, is_bipolar=True)
                return f"{meter}  {self.display_name} ({display_value:.2f})"
            else:
                # Show peak value within current chunk for mono
                meter = create_monitor_meter(self.peak_level, min_val, max_val, width=total_meter_width, color_scheme=color_scheme)
                return f"{meter}  {self.display_name} ({self.peak_level:.2f})"


class SequencerMonitor(NodeMonitor):
    """Custom monitor for sequencer nodes that shows timeline and current step."""
    
    def format_line(self) -> str:
        """Show sequencer timeline with current step indicator."""
        # Try to access sequencer-specific state
        current_step = getattr(self.node.state, 'current_step', None)
        
        # Get number of steps from model if available
        total_steps = None
        if hasattr(self.node, 'model'):
            # Use 'steps' field
            if hasattr(self.node.model, 'steps') and self.node.model.steps:
                total_steps = len(self.node.model.steps)
        
        if current_step is not None and total_steps is not None:
            # Get monitor range and color scheme from node (dynamically)
            min_val, max_val = getattr(self.node, '_monitor_range', (0.0, 1.0))
            color_scheme = getattr(self.node, '_monitor_color_scheme', 'level')
            is_bipolar = getattr(self, 'is_bipolar', False)
            
            # Create a narrower meter (about 1/3 of standard width = 8 chars)
            meter = create_monitor_meter(self.peak_level, min_val, max_val, width=8, color_scheme=color_scheme, is_bipolar=is_bipolar)
            
            # Build timeline visualization with | | | █ | | |
            # One character per step, with █ for current step
            timeline_chars = []
            for i in range(total_steps):
                if i == current_step:
                    timeline_chars.append('█')
                else:
                    timeline_chars.append('|')
            
            # Join with spaces for readability
            timeline = ' '.join(timeline_chars)
            
            # Format: "meter  timeline  name (step X/Y | value)"
            step_info = f"step {current_step + 1}/{total_steps}"
            return f"{meter}  {timeline}  {self.display_name} ({step_info} | {self.peak_level:.2f})"
        else:
            # Fallback to default display if we can't get sequencer info
            return super().format_line()


class SampleMonitor(NodeMonitor):
    """Custom monitor for sample nodes that shows audio waveform with playhead indicator."""
    
    def format_line(self) -> str:
        """Show sample waveform visualization with playhead position."""
        # Total width matches standard meter width
        total_width = 24
        
        # Get the audio buffer and playhead position
        if not hasattr(self.node, 'audio') or len(self.node.audio) == 0:
            return super().format_line()
        
        audio = self.node.audio
        audio_length = len(audio)
        playhead_pos = getattr(self.node.state, 'last_playhead_position', 0)
        
        # Calculate playhead position as fraction of buffer (0-1)
        playhead_fraction = (playhead_pos % audio_length) / audio_length if audio_length > 0 else 0
        
        # Build waveform visualization
        grey = '\033[90m'
        white = '\033[97m'
        reset = '\033[0m'
        
        # Unicode block characters for amplitude visualization (0-8 levels)
        # Use underscore for 0 so read head is always visible
        blocks = ['_', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']
        
        waveform_chars = []
        samples_per_char = max(1, audio_length // total_width)
        
        for i in range(total_width):
            # Calculate which section of the audio this character represents
            start_sample = i * samples_per_char
            end_sample = min(start_sample + samples_per_char, audio_length)
            
            if start_sample >= audio_length:
                waveform_chars.append(grey + '_' + reset)
                continue
            
            # Get peak amplitude for this section
            section = audio[start_sample:end_sample]
            if len(section) == 0:
                peak = 0
            else:
                peak = float(np.max(np.abs(section)))
            
            # Convert peak to block character index (0-8)
            block_idx = min(8, int(peak * 8))
            char = blocks[block_idx]
            
            # Determine if this character represents the playhead position
            char_start_fraction = i / total_width
            char_end_fraction = (i + 1) / total_width
            is_playhead = char_start_fraction <= playhead_fraction < char_end_fraction
            
            # Color: white for playhead, grey for others
            color = white if is_playhead else grey
            waveform_chars.append(color + char + reset)
        
        waveform = ''.join(waveform_chars)
        
        # Add position info
        position_info = f"pos: {int(playhead_pos)}/{audio_length}"
        return f"{waveform}  {self.display_name} ({position_info})"


class BufferMonitor(NodeMonitor):
    """Custom monitor for buffer nodes that shows buffer waveform with playhead indicator."""
    
    def format_line(self) -> str:
        """Show buffer waveform visualization with playhead/write-head position."""
        # Total width matches standard meter width
        total_width = 24
        
        # Get the buffer data
        if not hasattr(self.node, 'buffer_ref'):
            return super().format_line()
        
        buffer_data = self.node.buffer_ref['data']
        buffer_length = len(buffer_data)
        
        # Determine position based on mode
        if self.node.is_position_mode:
            # Position mode: use last_playhead_position
            position = getattr(self.node.state, 'last_playhead_position', 0)
        else:
            # Offset mode: use write_head
            position = self.node.buffer_ref.get('write_head', 0)
        
        # Calculate position as fraction of buffer (0-1)
        position_fraction = (position % buffer_length) / buffer_length if buffer_length > 0 else 0
        
        # Build waveform visualization
        grey = '\033[90m'
        white = '\033[97m'
        reset = '\033[0m'
        
        # Unicode block characters for amplitude visualization (0-8 levels)
        # Use underscore for 0 so read head is always visible
        blocks = ['_', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']
        
        waveform_chars = []
        samples_per_char = max(1, buffer_length // total_width)
        
        for i in range(total_width):
            # Calculate which section of the buffer this character represents
            start_sample = i * samples_per_char
            end_sample = min(start_sample + samples_per_char, buffer_length)
            
            if start_sample >= buffer_length:
                waveform_chars.append(grey + '_' + reset)
                continue
            
            # Get peak amplitude for this section
            section = buffer_data[start_sample:end_sample]
            if len(section) == 0:
                peak = 0
            else:
                peak = float(np.max(np.abs(section)))
            
            # Convert peak to block character index (0-8)
            block_idx = min(8, int(peak * 8))
            char = blocks[block_idx]
            
            # Determine if this character represents the position
            char_start_fraction = i / total_width
            char_end_fraction = (i + 1) / total_width
            is_position = char_start_fraction <= position_fraction < char_end_fraction
            
            # Color: white for position, grey for others
            color = white if is_position else grey
            waveform_chars.append(color + char + reset)
        
        waveform = ''.join(waveform_chars)
        
        # Add position info and mode
        mode = "pos" if self.node.is_position_mode else "wr"
        position_info = f"{mode}: {int(position)}/{buffer_length}"
        return f"{waveform}  {self.display_name} ({position_info})"


class MonitorRegistry:
    """
    Global registry for monitored nodes.
    Tracks which nodes should be displayed and provides formatted output.
    """
    
    def __init__(self):
        self.monitors: Dict[str, NodeMonitor] = {}
    
    def register(self, node_id: str, node: BaseNode):
        """Register a node for monitoring. Creates appropriate monitor type."""
        node_class_name = node.__class__.__name__
        
        # Check node type and create appropriate monitor
        if node_class_name in ('SequencerNode', 'AutomationNode'):
            self.monitors[node_id] = SequencerMonitor(node_id, node)
        elif node_class_name == 'SampleNode':
            self.monitors[node_id] = SampleMonitor(node_id, node)
        elif node_class_name == 'BufferNode':
            self.monitors[node_id] = BufferMonitor(node_id, node)
        else:
            # Default monitor for all other nodes
            self.monitors[node_id] = NodeMonitor(node_id, node)
    
    def unregister(self, node_id: str):
        """Remove a node from monitoring."""
        if node_id in self.monitors:
            del self.monitors[node_id]
    
    def update(self, node_id: str, output: np.ndarray):
        """Update a monitored node with its latest output."""
        if node_id in self.monitors:
            self.monitors[node_id].update(output)
    
    def get_display_lines(self) -> list[str]:
        """Get formatted display lines for all monitored nodes."""
        if not self.monitors:
            return []
        
        # Sort by node_id for consistent ordering
        sorted_monitors = sorted(self.monitors.values(), key=lambda m: m.node_id)
        return [monitor.format_line() for monitor in sorted_monitors]
    
    def clear(self):
        """Clear all monitors (useful for hot reload)."""
        self.monitors.clear()


# Global singleton instance
_monitor_registry = MonitorRegistry()


def get_monitor_registry() -> MonitorRegistry:
    """Get the global monitor registry instance."""
    return _monitor_registry
