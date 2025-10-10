from __future__ import annotations
from enum import Enum
import numpy as np
import scipy.signal
from pydantic import ConfigDict
from config import SAMPLE_RATE
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.oscillator import OSCILLATOR_RENDER_ARGS
from nodes.wavable_value import WavableValue, wavable_value_node_factory

class FilterTypes(str, Enum):
    HIGHPASS = "HIGHPASS"
    LOWPASS = "LOWPASS"
    HIGH = "HIGHPASS"
    LOW = "LOWPASS"
    HP = "HIGHPASS"
    LP = "LOWPASS"
    HPF = "HIGHPASS"
    LPF = "LOWPASS"
    BPF = "BANDPASS"
    BANDPASS = "BANDPASS"
    BAND = "BANDPASS"

class FilterModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    cutoff: WavableValue  # Cutoff frequency in Hz
    peak: WavableValue = 0.0  # A value between -1 and 1 that translates to a 0.5 to 50 Q factor
    type: str = "lowpass"
    signal: BaseNodeModel = None

class FilterNode(BaseNode):
    def __init__(self, model: FilterModel):
        from nodes.node_utils.instantiate_node import instantiate_node
        super().__init__(model)
        self.model = model
        self.cutoff_node = wavable_value_node_factory(model.cutoff)
        self.peak_node = wavable_value_node_factory(model.peak)
        self.signal_node = instantiate_node(model.signal)
        
        # Filter state variables for continuity between chunks
        self.x1 = 0.0
        self.x2 = 0.0
        self.y1 = 0.0
        self.y2 = 0.0
        
        # State for scipy.signal.lfilter (used when cutoff is constant)
        self.zi = None  # Filter initial conditions

    def render(self, num_samples=None, **params):
        super().render(num_samples)
        
        # If num_samples is None, get the full child signal
        if num_samples is None:
            num_samples = self.resolve_num_samples(num_samples)
            if num_samples is None:
                # Need to get full signal from child
                signal_wave = self.render_full_child_signal(self.signal_node, **self.get_params_for_children(params))
                if len(signal_wave) == 0:
                    return np.array([])
                
                num_samples = len(signal_wave)
                return self._apply_filter(signal_wave, num_samples, params)
        
        signal_wave = self.signal_node.render(num_samples, **self.get_params_for_children(params))
        
        # If signal is done, we're done
        if len(signal_wave) == 0:
            return np.array([], dtype=np.float32)
        
        return self._apply_filter(signal_wave, num_samples, params)
    
    def _apply_filter(self, signal_wave, num_samples, params):
        """Apply filter to the signal wave"""
        cutoff = self.cutoff_node.render(num_samples, **self.get_params_for_children(params, OSCILLATOR_RENDER_ARGS))
        
        # Render only a single sample from peak to get current value
        peak_wave = self.peak_node.render(1, **self.get_params_for_children(params, OSCILLATOR_RENDER_ARGS))
        peak = peak_wave[0] if len(peak_wave) > 0 else 0.0

        if len(cutoff) == 1:
            cutoff = cutoff[0]

        filter_type = self.model.type.lower()
        q = normalized_to_q(peak)

        if np.isscalar(cutoff):
            if filter_type == "lowpass":
                b, a = biquad_lowpass(cutoff, q, SAMPLE_RATE)
            elif filter_type == "highpass":
                b, a = biquad_highpass(cutoff, q, SAMPLE_RATE)
            elif filter_type == "bandpass":
                nyq = SAMPLE_RATE / 2.0
                b, a = scipy.signal.iirpeak(cutoff / nyq, Q=q)
            else:
                raise ValueError(f"Unsupported filter type: {filter_type}")
            
            # Initialize filter state if needed
            if self.zi is None:
                self.zi = scipy.signal.lfilter_zi(b, a)
            
            # Apply filter with state
            filtered, self.zi = scipy.signal.lfilter(b, a, signal_wave, zi=self.zi)
            return filtered
        else:
            # Modulated cutoff with Q support
            out = np.zeros_like(signal_wave)
            
            # Use instance state variables for continuity between chunks
            x1, x2 = self.x1, self.x2
            y1, y2 = self.y1, self.y2

            for i in range(len(signal_wave)):
                fc = cutoff[i]

                if filter_type == "lowpass":
                    b, a = biquad_lowpass(fc, q, SAMPLE_RATE)
                elif filter_type == "highpass":
                    b, a = biquad_highpass(fc, q, SAMPLE_RATE)
                elif filter_type == "bandpass":
                    b, a = biquad_bandpass(fc, q, SAMPLE_RATE)
                else:
                    raise ValueError(f"Modulated filter type not supported: {filter_type}")

                # Direct Form I Biquad filter (per-sample, using past values)
                x0 = signal_wave[i]
                y0 = b[0] * x0 + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2

                out[i] = y0
                x2, x1 = x1, x0
                y2, y1 = y1, y0

            # Store state for next chunk
            self.x1, self.x2 = x1, x2
            self.y1, self.y2 = y1, y2

            return out

def normalized_to_q(n: float) -> float:
    # Map n ∈ [-1, 1] → Q ∈ [0.5, 50] with smooth curve
    min_q = 0.5
    max_q = 50
    neutral_q = 0.707

    # Convert n ∈ [-1, 1] to t ∈ [0, 1]
    t = (n + 1) / 2

    # Use a smooth non-linear curve (e.g., exponential base)
    # Curve centered at t=0.5 → neutral Q
    curved = np.interp(t, [0, 0.5, 1], [min_q, neutral_q, max_q])
    return curved

def biquad_lowpass(fc: float, q: float, fs: float):
    w0 = 2 * np.pi * fc / fs
    alpha = np.sin(w0) / (2 * q)

    cos_w0 = np.cos(w0)
    b0 = (1 - cos_w0) / 2
    b1 = 1 - cos_w0
    b2 = (1 - cos_w0) / 2
    a0 = 1 + alpha
    a1 = -2 * cos_w0
    a2 = 1 - alpha

    return normalize_biquad(b0, b1, b2, a0, a1, a2)

def biquad_highpass(fc: float, q: float, fs: float):
    w0 = 2 * np.pi * fc / fs
    alpha = np.sin(w0) / (2 * q)

    cos_w0 = np.cos(w0)
    b0 = (1 + cos_w0) / 2
    b1 = -(1 + cos_w0)
    b2 = (1 + cos_w0) / 2
    a0 = 1 + alpha
    a1 = -2 * cos_w0
    a2 = 1 - alpha

    return normalize_biquad(b0, b1, b2, a0, a1, a2)

def biquad_bandpass(fc: float, q: float, fs: float):
    w0 = 2 * np.pi * fc / fs
    alpha = np.sin(w0) / (2 * q)

    cos_w0 = np.cos(w0)
    b0 = alpha
    b1 = 0
    b2 = -alpha
    a0 = 1 + alpha
    a1 = -2 * cos_w0
    a2 = 1 - alpha

    return normalize_biquad(b0, b1, b2, a0, a1, a2)

def normalize_biquad(b0, b1, b2, a0, a1, a2):
    return (
        np.array([b0, b1, b2]) / a0,
        np.array([1.0, a1 / a0, a2 / a0])
    )

FILTER_DEFINITION = NodeDefinition("filter", FilterNode, FilterModel)
