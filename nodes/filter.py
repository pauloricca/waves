from __future__ import annotations
from enum import Enum
import numpy as np
import scipy.signal
from pydantic import ConfigDict
from config import SAMPLE_RATE
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.wavable_value import WavableValue
from utils import empty_mono

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
    def __init__(self, model: FilterModel, node_id: str, state, do_initialise_state=True):
        super().__init__(model, node_id, state, do_initialise_state)
        self.model = model
        self.cutoff_node = self.instantiate_child_node(model.cutoff, "cutoff")
        self.peak_node = self.instantiate_child_node(model.peak, "peak")
        self.signal_node = self.instantiate_child_node(model.signal, "signal")
        
        # Persistent state for continuity between chunks (survives hot reload)
        if do_initialise_state:
            self.state.x1 = 0.0
            self.state.x2 = 0.0
            self.state.y1 = 0.0
            self.state.y2 = 0.0
            # Coefficient caching
            self.state.last_cutoff = None
            self.state.last_peak = None
            self.state.cached_b = None
            self.state.cached_a = None

    def _do_render(self, num_samples=None, context=None, num_channels=1, **params):
        # If num_samples is None, get the full child signal
        if num_samples is None:
            num_samples = self.resolve_num_samples(num_samples)
            if num_samples is None:
                # Need to get full signal from child
                signal_wave = self.render_full_child_signal(self.signal_node, context, **self.get_params_for_children(params))
                if len(signal_wave) == 0:
                    return np.array([])
                
                num_samples = len(signal_wave)
                return self._apply_filter(signal_wave, num_samples, context, params)
        
        signal_wave = self.signal_node.render(num_samples, context, **self.get_params_for_children(params))
        
        # If signal is done, we're done
        if len(signal_wave) == 0:
            return empty_mono()
        
        return self._apply_filter(signal_wave, num_samples, context, params)
    
    def _apply_filter(self, signal_wave, num_samples, context, params):
        """Apply filter to the signal wave"""
        cutoff = self.cutoff_node.render(num_samples, context, **self.get_params_for_children(params))
        
        # Render only a single sample from peak to get current value
        peak_wave = self.peak_node.render(1, context, **self.get_params_for_children(params))
        peak = peak_wave[0] if len(peak_wave) > 0 else 0.0

        if len(cutoff) == 1:
            cutoff = cutoff[0]

        filter_type = self.model.type.lower()
        q = normalized_to_q(peak)
        
        # Clamp cutoff to valid range (20 Hz to Nyquist - 100 Hz)
        nyquist = SAMPLE_RATE / 2.0
        min_cutoff = 20.0
        max_cutoff = nyquist - 100.0
        
        if np.isscalar(cutoff):
            cutoff = np.clip(cutoff, min_cutoff, max_cutoff)
        else:
            cutoff = np.clip(cutoff, min_cutoff, max_cutoff)

        if np.isscalar(cutoff):
            # Static cutoff - use cached coefficients if parameters haven't changed
            params_changed = (self.state.last_cutoff != cutoff or 
                             self.state.last_peak != peak or 
                             self.state.cached_b is None)
            
            if params_changed:
                if filter_type == "lowpass":
                    self.state.cached_b, self.state.cached_a = biquad_lowpass(cutoff, q, SAMPLE_RATE)
                elif filter_type == "highpass":
                    self.state.cached_b, self.state.cached_a = biquad_highpass(cutoff, q, SAMPLE_RATE)
                elif filter_type == "bandpass":
                    self.state.cached_b, self.state.cached_a = biquad_bandpass(cutoff, q, SAMPLE_RATE)
                else:
                    raise ValueError(f"Unsupported filter type: {filter_type}")
                
                self.state.last_cutoff = cutoff
                self.state.last_peak = peak
            
            # Apply filter using simple biquad (no scipy.signal.lfilter)
            return self._apply_biquad_simple(signal_wave, self.state.cached_b, self.state.cached_a)
        else:
            # Modulated cutoff - check if it's actually changing significantly
            cutoff_change = np.max(np.abs(np.diff(cutoff))) if len(cutoff) > 1 else 0
            
            # If cutoff is barely changing (e.g., MIDI CC at steady value), treat as static
            # This threshold means less than 1 Hz change across the buffer
            if cutoff_change < 1.0:
                avg_cutoff = np.mean(cutoff)
                
                # Check if we should recalculate coefficients
                params_changed = (self.state.last_cutoff is None or 
                                 abs(self.state.last_cutoff - avg_cutoff) > 0.5 or
                                 self.state.last_peak != peak or
                                 self.state.cached_b is None)
                
                if params_changed:
                    if filter_type == "lowpass":
                        self.state.cached_b, self.state.cached_a = biquad_lowpass(avg_cutoff, q, SAMPLE_RATE)
                    elif filter_type == "highpass":
                        self.state.cached_b, self.state.cached_a = biquad_highpass(avg_cutoff, q, SAMPLE_RATE)
                    elif filter_type == "bandpass":
                        self.state.cached_b, self.state.cached_a = biquad_bandpass(avg_cutoff, q, SAMPLE_RATE)
                    else:
                        raise ValueError(f"Unsupported filter type: {filter_type}")
                    
                    self.state.last_cutoff = avg_cutoff
                    self.state.last_peak = peak
                
                return self._apply_biquad_simple(signal_wave, self.state.cached_b, self.state.cached_a)
            
            # Truly modulated cutoff - use per-sample processing
            out = np.zeros_like(signal_wave)
            
            # Use instance state variables for continuity between chunks
            x1, x2 = self.state.x1, self.state.x2
            y1, y2 = self.state.y1, self.state.y2

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
            self.state.x1, self.state.x2 = x1, x2
            self.state.y1, self.state.y2 = y1, y2

            return out
    
    def _apply_biquad_simple(self, signal_wave, b, a):
        """Apply biquad filter using simple state variables (no scipy zi)"""
        out = np.zeros_like(signal_wave)
        
        # Use instance state variables for continuity between chunks
        x1, x2 = self.state.x1, self.state.x2
        y1, y2 = self.state.y1, self.state.y2

        for i in range(len(signal_wave)):
            # Direct Form I Biquad filter
            x0 = signal_wave[i]
            y0 = b[0] * x0 + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2

            out[i] = y0
            x2, x1 = x1, x0
            y2, y1 = y1, y0

        # Store state for next chunk
        self.state.x1, self.state.x2 = x1, x2
        self.state.y1, self.state.y2 = y1, y2

        return out

def normalized_to_q(n: float) -> float:
    # Map n ∈ [-1, 1] → Q ∈ [0.5, 10] with smooth curve
    # More conservative max Q to avoid extreme resonance
    min_q = 0.5
    max_q = 10.0  # Reduced from 50 - much more musical range
    neutral_q = 0.707

    # Convert n ∈ [-1, 1] to t ∈ [0, 1]
    t = (n + 1) / 2
    
    # Use exponential curve for more control at lower Q values
    # This gives finer control in the musical range (Q < 5)
    if t < 0.5:
        # Map 0->0.5 to min_q->neutral_q linearly
        curved = min_q + (neutral_q - min_q) * (t / 0.5)
    else:
        # Map 0.5->1 to neutral_q->max_q with exponential curve
        # This compresses the higher Q values
        normalized = (t - 0.5) / 0.5  # 0 to 1
        curved = neutral_q + (max_q - neutral_q) * (normalized ** 2)
    
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

    b, a = normalize_biquad(b0, b1, b2, a0, a1, a2)
    
    # Apply gain compensation for resonance
    # At high Q, the filter boosts at the cutoff frequency
    # Compensate by reducing overall gain
    if q > 1.0:
        gain_compensation = 1.0 / np.sqrt(q)
        b = b * gain_compensation
    
    return b, a

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

    b, a = normalize_biquad(b0, b1, b2, a0, a1, a2)
    
    # Apply gain compensation for resonance
    if q > 1.0:
        gain_compensation = 1.0 / np.sqrt(q)
        b = b * gain_compensation
    
    return b, a

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

    b, a = normalize_biquad(b0, b1, b2, a0, a1, a2)
    
    # Bandpass already has more controlled gain, but still compensate
    if q > 1.0:
        gain_compensation = 1.0 / np.sqrt(q)
        b = b * gain_compensation
    
    return b, a

def normalize_biquad(b0, b1, b2, a0, a1, a2):
    return (
        np.array([b0, b1, b2]) / a0,
        np.array([1.0, a1 / a0, a2 / a0])
    )

FILTER_DEFINITION = NodeDefinition("filter", FilterNode, FilterModel)
