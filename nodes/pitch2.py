"""Pitch detection node using the YIN algorithm.

The existing :mod:`pitch` node relies on a simple autocorrelation peak
search that can become unstable for noisier or polyphonic material.  This
node keeps a similar interface but replaces the detector with a small and
efficient implementation of the YIN cumulative mean normalized difference
function (CMNDF).  The CMNDF is robust against amplitude changes and finds a
stable minimum corresponding to the fundamental period.

The algorithm is evaluated on a short trailing window of the incoming audio
to keep the computational load low while still tracking pitch in real time.
"""

from typing import Optional

import numpy as np
from pydantic import ConfigDict

from config import SAMPLE_RATE
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.wavable_value import WavableValue


class Pitch2NodeModel(BaseNodeModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    signal: WavableValue
    min_freq: float = 80.0
    max_freq: float = 1000.0
    per_sample: bool = False
    smoothing: float = 0.0
    yin_threshold: float = 0.15  # Lower = stricter confidence requirement


class Pitch2Node(BaseNode):
    """Pitch detector using the YIN CMNDF algorithm."""

    def __init__(
        self,
        model: Pitch2NodeModel,
        node_id: str,
        state: Optional[object] = None,
        do_initialise_state: bool = True,
    ) -> None:
        super().__init__(model, node_id, state, do_initialise_state)
        self.model = model

        if do_initialise_state:
            self.state.last_detected_freq = 0.0
            self.state.buffer = np.array([], dtype=np.float32)

        self.signal_node = self.instantiate_child_node(model.signal, "signal")

        # Convert the frequency range to period bounds in samples.
        self.max_period = int(SAMPLE_RATE / model.min_freq)
        self.min_period = max(1, int(SAMPLE_RATE / model.max_freq))

    def _detect_pitch_yin(self, signal: np.ndarray) -> float:
        """Detect the fundamental frequency using the YIN method."""

        if len(signal) < self.max_period * 2:
            # Not enough context to perform the analysis reliably.
            return self.state.last_detected_freq

        window_size = min(len(signal), self.max_period * 4)
        windowed = signal[-window_size:].astype(np.float32, copy=False)

        # Remove DC and apply a lightweight window to tame edges.
        windowed = windowed - np.mean(windowed)
        windowed *= np.hanning(len(windowed)).astype(np.float32)

        tau_max = self.max_period
        tau_min = self.min_period

        # Difference function d(tau).
        diff = np.zeros(tau_max, dtype=np.float32)
        for tau in range(1, tau_max):
            delayed = windowed[tau:]
            ahead = windowed[:-tau]
            diff[tau] = np.sum((ahead - delayed) ** 2)

        # Cumulative mean normalized difference function (CMNDF).
        cmndf = np.zeros_like(diff)
        cumulative_sum = 0.0
        for tau in range(1, tau_max):
            cumulative_sum += diff[tau]
            if cumulative_sum == 0:
                cmndf[tau] = 1.0
            else:
                cmndf[tau] = diff[tau] * tau / cumulative_sum

        # Find the first tau that falls below the threshold.
        candidates = np.where(cmndf[tau_min:tau_max] < self.model.yin_threshold)[0]

        if len(candidates) == 0:
            # If nothing met the threshold, choose the best local minimum.
            tau = int(np.argmin(cmndf[tau_min:tau_max]) + tau_min)
            if cmndf[tau] >= 1.0:
                return 0.0
        else:
            tau = int(candidates[0] + tau_min)
            # Local parabolic refinement if the following sample improves.
            while tau + 1 < tau_max and cmndf[tau + 1] < cmndf[tau]:
                tau += 1

        if tau <= 0:
            return 0.0

        # Parabolic interpolation around the minimum for sub-sample accuracy.
        if 1 <= tau < tau_max - 1:
            s0, s1, s2 = cmndf[tau - 1], cmndf[tau], cmndf[tau + 1]
            denominator = (2 * s1) - s0 - s2
            if denominator != 0:
                tau = tau + 0.5 * (s0 - s2) / denominator

        frequency = SAMPLE_RATE / tau if tau > 0 else 0.0
        if not np.isfinite(frequency) or frequency <= 0:
            return 0.0

        return frequency

    def _do_render(self, num_samples=None, context=None, num_channels=1, **params):
        input_signal = self.signal_node.render(num_samples, context, num_channels, **params)

        if len(input_signal) == 0:
            return np.array([])

        # Convert to mono for analysis.
        if input_signal.ndim == 2:
            input_signal = np.mean(input_signal, axis=1)

        self.state.buffer = np.concatenate([self.state.buffer, input_signal])

        max_buffer_size = self.max_period * 5
        if len(self.state.buffer) > max_buffer_size:
            self.state.buffer = self.state.buffer[-max_buffer_size:]

        if self.model.per_sample:
            output = np.zeros(len(input_signal), dtype=np.float32)
            for i in range(len(input_signal)):
                if i < len(input_signal) - 1:
                    analysis_window = self.state.buffer[: -(len(input_signal) - i - 1)]
                else:
                    analysis_window = self.state.buffer

                detected = self._detect_pitch_yin(analysis_window)

                if self.model.smoothing > 0:
                    detected = (
                        self.state.last_detected_freq * self.model.smoothing
                        + detected * (1 - self.model.smoothing)
                    )

                self.state.last_detected_freq = detected
                output[i] = detected
        else:
            detected = self._detect_pitch_yin(self.state.buffer)

            if self.model.smoothing > 0:
                detected = (
                    self.state.last_detected_freq * self.model.smoothing
                    + detected * (1 - self.model.smoothing)
                )

            self.state.last_detected_freq = detected
            output = np.full(len(input_signal), detected, dtype=np.float32)

        return output


PITCH2_DEFINITION = NodeDefinition(
    name="pitch2",
    model=Pitch2NodeModel,
    node=Pitch2Node,
)

