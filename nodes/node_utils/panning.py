"""
Panning utilities for stereo audio processing.
"""

import numpy as np
from typing import Union


def apply_panning(mono_signal: np.ndarray, pan_value: Union[float, np.ndarray]) -> np.ndarray:
    """
    Apply equal-power panning to mono signal to create stereo output.
    
    Equal-power panning ensures constant perceived loudness as sounds move
    across the stereo field. Uses trigonometric panning law where at center
    position, both channels are at sqrt(2)/2 ≈ 0.707 amplitude.
    
    Args:
        mono_signal: 1D array of mono audio samples
        pan_value: Pan position(s) from -1 (full left) to 1 (full right).
                  Can be a scalar for static panning or 1D array for dynamic panning.
                  0 = center position.
    
    Returns:
        2D array of shape (num_samples, 2) with stereo audio
        
    Examples:
        # Static center panning
        stereo = apply_panning(mono, 0)
        
        # Static hard left
        stereo = apply_panning(mono, -1)
        
        # Dynamic panning
        pan_lfo = np.sin(np.linspace(0, 2*np.pi, len(mono)))  # -1 to 1
        stereo = apply_panning(mono, pan_lfo)
    """
    # Handle special cases for efficiency (no calculation needed)
    is_scalar_pan = not isinstance(pan_value, np.ndarray)
    
    if is_scalar_pan:
        if pan_value == -1:
            # Full left
            return np.stack([mono_signal, np.zeros_like(mono_signal)], axis=-1)
        elif pan_value == 1:
            # Full right
            return np.stack([np.zeros_like(mono_signal), mono_signal], axis=-1)
        elif pan_value == 0:
            # Center - equal power means sqrt(2)/2 in each channel
            return np.stack([mono_signal * 0.7071067811865476, 
                           mono_signal * 0.7071067811865476], axis=-1)
    
    # General case: equal-power panning
    # Convert pan from [-1, 1] to [0, 1]
    pan_normalized = (pan_value + 1) / 2
    
    # Equal power law using trigonometric functions
    # At center, both channels are at sqrt(2)/2 ≈ 0.707
    angle = pan_normalized * np.pi / 2
    left_gain = np.cos(angle)
    right_gain = np.sin(angle)
    
    # Apply gains
    left_channel = mono_signal * left_gain
    right_channel = mono_signal * right_gain
    
    return np.stack([left_channel, right_channel], axis=-1)
