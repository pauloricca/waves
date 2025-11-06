"""
Pitch detection node using autocorrelation method.

Analyzes an input signal and outputs the detected fundamental frequency.
The output can vary per-sample for accurate pitch tracking, or be constant per chunk.

Parameters:
- signal: The input audio signal to analyze
- min_freq: Minimum frequency to detect (default: 80 Hz, low E on bass)
- max_freq: Maximum frequency to detect (default: 1000 Hz, high C6)
- per_sample: If True, update pitch every sample (slower but smoother). If False, detect once per chunk (faster)
- smoothing: How much to smooth the output (0 = no smoothing, 0.99 = heavy smoothing)
"""

import numpy as np
from typing import Optional
from pydantic import ConfigDict

from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.wavable_value import WavableValue
from config import SAMPLE_RATE


class PitchNodeModel(BaseNodeModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    signal: WavableValue
    min_freq: float = 80.0  # Minimum detectable frequency (Hz)
    max_freq: float = 1000.0  # Maximum detectable frequency (Hz)
    per_sample: bool = False  # If True, detect pitch per sample; if False, per chunk
    smoothing: float = 0.0  # Exponential smoothing factor (0-0.99)


class PitchNode(BaseNode):
    def __init__(self, model: PitchNodeModel, node_id: str, state=None, do_initialise_state=True):
        super().__init__(model, node_id, state, do_initialise_state)
        self.model = model
        
        if do_initialise_state:
            self.state.last_detected_freq = 0.0
            self.state.buffer = np.array([])  # Buffer for maintaining context between chunks
        
        self.signal_node = self.instantiate_child_node(model.signal, "signal")
        
        # Calculate min/max period in samples based on frequency range
        self.max_period = int(SAMPLE_RATE / model.min_freq)
        self.min_period = int(SAMPLE_RATE / model.max_freq)
        
    def _detect_pitch_autocorrelation(self, signal: np.ndarray) -> float:
        """
        Detect pitch using autocorrelation method.
        Returns frequency in Hz, or 0 if no clear pitch detected.
        """
        if len(signal) < self.max_period * 2:
            # Not enough samples for reliable detection
            return self.state.last_detected_freq
        
        # Use only the most recent samples for detection (window size)
        window_size = min(len(signal), self.max_period * 3)
        windowed = signal[-window_size:]
        
        # Apply Hanning window to reduce edge effects
        window = np.hanning(len(windowed))
        windowed = windowed * window
        
        # Compute autocorrelation using numpy
        autocorr = np.correlate(windowed, windowed, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Keep only positive lags
        
        # Normalize
        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]
        else:
            return 0.0
        
        # Find the first peak after the minimum period
        # Start search after min_period to avoid detecting harmonics
        search_range = autocorr[self.min_period:self.max_period]
        
        if len(search_range) == 0:
            return 0.0
        
        # Find local maxima
        peak_idx = np.argmax(search_range)
        peak_value = search_range[peak_idx]
        
        # Require a minimum correlation threshold for confidence
        if peak_value < 0.3:  # Threshold for "clear" pitch
            return 0.0
        
        # Convert period to frequency
        period = peak_idx + self.min_period
        frequency = SAMPLE_RATE / period
        
        return frequency
    
    def _do_render(self, num_samples=None, context=None, num_channels=1, **params):
        # Get input signal
        input_signal = self.signal_node.render(num_samples, context, num_channels, **params)
        
        if len(input_signal) == 0:
            return np.array([])
        
        # Convert stereo to mono if needed (average channels)
        if input_signal.ndim == 2:
            input_signal = np.mean(input_signal, axis=1)
        
        # Append to buffer for continuous analysis
        self.state.buffer = np.concatenate([self.state.buffer, input_signal])
        
        # Keep buffer size manageable (keep last N samples for context)
        max_buffer_size = self.max_period * 4
        if len(self.state.buffer) > max_buffer_size:
            self.state.buffer = self.state.buffer[-max_buffer_size:]
        
        if self.model.per_sample:
            # Detect pitch for each sample (expensive but smooth)
            output = np.zeros(len(input_signal))
            for i in range(len(input_signal)):
                # Use buffer up to current position
                analysis_window = self.state.buffer[:-(len(input_signal)-i-1)] if i < len(input_signal)-1 else self.state.buffer
                detected = self._detect_pitch_autocorrelation(analysis_window)
                
                # Apply smoothing
                if self.model.smoothing > 0:
                    detected = self.state.last_detected_freq * self.model.smoothing + detected * (1 - self.model.smoothing)
                
                self.state.last_detected_freq = detected
                output[i] = detected
        else:
            # Detect pitch once per chunk (fast)
            detected = self._detect_pitch_autocorrelation(self.state.buffer)
            
            # Apply smoothing
            if self.model.smoothing > 0:
                detected = self.state.last_detected_freq * self.model.smoothing + detected * (1 - self.model.smoothing)
            
            self.state.last_detected_freq = detected
            
            # Return constant value for the whole chunk
            output = np.full(len(input_signal), detected)
        
        return output


PITCH_DEFINITION = NodeDefinition(
    name="pitch",
    model=PitchNodeModel,
    node=PitchNode,
)
