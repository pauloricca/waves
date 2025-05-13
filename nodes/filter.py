from __future__ import annotations
from enum import Enum
import numpy as np
import scipy.signal
from pydantic import ConfigDict, field_validator
from config import SAMPLE_RATE
from models.models import BaseNodeModel
from nodes.node_utils.base import BaseNode
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.wavable_value import WavableValue, WavableValueNode

class FilterTypes(str, Enum):
    HIGHPASS = "HIGHPASS"
    LOWPASS = "LOWPASS"
    HIGH = "HIGHPASS"
    LOW = "LOWPASS"
    HP = "HIGHPASS"
    LP = "LOWPASS"
    HPF = "HIGHPASS"
    LPF = "LOWPASS"

class FilterModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    cutoff: WavableValue  # Cutoff frequency in Hz
    peak: float = 0.0  # A value between -1 and 1 that translates to a 0.5 to 50 Q factor
    type: FilterTypes = FilterTypes.LOWPASS
    signal: BaseNodeModel = None

    @field_validator("type", mode="before")
    @classmethod
    def normalize_wave_type(cls, v):
        if v is None:
            return FilterTypes.LOWPASS.value
        if isinstance(v, str):
            v = v.upper()
            if v in FilterTypes.__members__:
                return FilterTypes[v].value
        return v

class FilterNode(BaseNode):
    def __init__(self, filter_model: FilterModel):
        from nodes.node_utils.instantiate_node import instantiate_node
        self.filter_model = filter_model
        self.cutoff_node = WavableValueNode(filter_model.cutoff)
        self.signal_node = instantiate_node(filter_model.signal)

    def render(self, num_samples, **kwargs):
        wave = self.signal_node.render(num_samples, **kwargs)
        cutoff = self.cutoff_node.render(len(wave))
        if len(cutoff) == 1:
            cutoff = cutoff[0]

        filter_type = self.filter_model.type.lower()
        q = normalized_to_q(self.filter_model.peak)

        if np.isscalar(cutoff):
            if filter_type == "lowpass":
                b, a = biquad_lowpass(cutoff, q, SAMPLE_RATE)
            elif filter_type == "highpass":
                b, a = biquad_highpass(cutoff, q, SAMPLE_RATE)
            elif filter_type == "bandpass":
                nyq = SAMPLE_RATE / 2.0
                b, a = scipy.signal.iirpeak(cutoff / nyq, Q=q)

            return scipy.signal.lfilter(b, a, wave)
        else:
            # Blockwise filter application for efficiency
            block_size = 16
            out = np.zeros_like(wave)
            for start in range(0, len(wave), block_size):
                end = min(start + block_size, len(wave))
                avg_fc = np.mean(cutoff[start:end])
                if filter_type == "lowpass":
                    b, a = biquad_lowpass(avg_fc, q, SAMPLE_RATE)
                elif filter_type == "highpass":
                    b, a = biquad_highpass(avg_fc, q, SAMPLE_RATE)

                out[start:end] = scipy.signal.lfilter(b, a, wave[start:end])

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

def normalize_biquad(b0, b1, b2, a0, a1, a2):
    return (
        np.array([b0, b1, b2]) / a0,
        np.array([1.0, a1 / a0, a2 / a0])
    )

FILTER_DEFINITION = NodeDefinition("filter", FilterNode, FilterModel)
