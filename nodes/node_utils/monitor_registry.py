"""
Monitor registry for displaying real-time node output visualization.
Tracks monitored nodes and provides formatted display lines for the visualizer.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Dict
import numpy as np

if TYPE_CHECKING:
    from nodes.node_utils.base_node import BaseNode


def create_monitor_meter(value: float, min_val: float, max_val: float, width: int = 40, color_scheme: str = "level") -> str:
    """
    Create a visual meter using vertical bars (reuses display_stats logic).
    
    Args:
        value: Current value
        min_val: Minimum value of the range
        max_val: Maximum value of the range
        width: Width of the meter in characters
        color_scheme: "level" for green-yellow-red, "value" for blue
    
    Returns:
        A string representing the meter with colors
    """
    # Normalize value to 0-1 range
    if max_val == min_val:
        normalized = 0.5  # Default to middle if no range
    else:
        normalized = (value - min_val) / (max_val - min_val)
        normalized = max(0.0, min(1.0, normalized))  # Clamp to 0-1
    
    # Calculate how many blocks to fill
    filled = normalized * width
    full_blocks = int(filled)
    partial = filled - full_blocks
    
    # Choose partial block character based on fractional part
    # Unicode block elements: ▏▎▍▌▋▊▉█
    partial_chars = ['', '▏', '▎', '▍', '▌', '▋', '▊', '▉']
    partial_idx = int(partial * len(partial_chars))
    partial_char = partial_chars[partial_idx] if partial_idx < len(partial_chars) else ''
    
    # Build the meter
    # Choose color scheme
    if color_scheme == "value":
        # Blue for value-based meters (not heat/level based)
        color_code = '\033[94m'  # Blue
    else:
        # Level-based: Green (0-0.7), Yellow (0.7-0.9), Red (0.9+)
        if normalized < 0.7:
            color_code = '\033[92m'  # Green
        elif normalized < 0.9:
            color_code = '\033[93m'  # Yellow
        else:
            color_code = '\033[91m'  # Red
    
    reset_code = '\033[0m'
    grey_code = '\033[90m'
    
    # Full blocks
    filled_part = '█' * full_blocks + partial_char
    # Empty part with vertical bars
    empty_part = f"{grey_code}{'|' * (width - full_blocks - (1 if partial_char else 0))}{reset_code}"
    
    return f"{color_code}{filled_part}{reset_code}{empty_part}"


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
        
        # Check if output is stereo (2D array with 2 channels)
        if output.ndim == 2 and output.shape[1] == 2:
            if use_abs:
                self.peak_left = float(np.max(np.abs(output[:, 0])))
                self.peak_right = float(np.max(np.abs(output[:, 1])))
            else:
                self.peak_left = float(np.max(output[:, 0]))
                self.peak_right = float(np.max(output[:, 1]))
            self.peak_level = max(self.peak_left, self.peak_right)
        else:
            if use_abs:
                self.peak_level = float(np.max(np.abs(output)))
            else:
                self.peak_level = float(np.max(output))
    
    def format_line(self) -> str:
        """Format display line for this monitor. Override in subclasses for custom display."""
        # Get monitor range and color scheme from node (dynamically, so it picks up changes)
        min_val, max_val = getattr(self.node, '_monitor_range', (0.0, 1.0))
        color_scheme = getattr(self.node, '_monitor_color_scheme', 'level')
        
        # Total width for meters (slightly wider than main output meter)
        total_meter_width = 24
        
        if self.is_stereo:
            # Show separate L/R meters for stereo
            # Total: "L " (2) + meter (10) + " R " (3) + meter (9) = 24 chars
            meter_l = create_monitor_meter(self.peak_left, min_val, max_val, width=10, color_scheme=color_scheme)
            meter_r = create_monitor_meter(self.peak_right, min_val, max_val, width=9, color_scheme=color_scheme)
            value_str = f"L:{self.peak_left:.2f} R:{self.peak_right:.2f}"
            return f"L {meter_l} R {meter_r}  {self.display_name} ({value_str})"
        else:
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
            
            # Create a narrower meter (about 1/3 of standard width = 8 chars)
            meter = create_monitor_meter(self.peak_level, min_val, max_val, width=8, color_scheme=color_scheme)
            
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


class MonitorRegistry:
    """
    Global registry for monitored nodes.
    Tracks which nodes should be displayed and provides formatted output.
    """
    
    def __init__(self):
        self.monitors: Dict[str, NodeMonitor] = {}
    
    def register(self, node_id: str, node: BaseNode):
        """Register a node for monitoring. Creates appropriate monitor type."""
        # Check if this is a sequencer or automation node (both have step-based structure)
        if node.__class__.__name__ in ('SequencerNode', 'AutomationNode'):
            self.monitors[node_id] = SequencerMonitor(node_id, node)
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
