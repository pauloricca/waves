    
from __future__ import annotations
import numpy as np
from config import SAMPLE_RATE
from models.models import DelayModel
from nodes.base_node import BaseNode
from nodes.instantiate_node import instantiate_node


class DelayNode(BaseNode):
    def __init__(self, delay_model: DelayModel):
        self.delay_model = delay_model
        self.signal_node = instantiate_node(delay_model.signal)

    def render(self, num_samples):
        wave = self.signal_node.render(num_samples)
        n_delay_time_samples = int(SAMPLE_RATE * self.delay_model.time)
        delayed_wave = np.zeros(len(wave) + n_delay_time_samples * self.delay_model.repeats)

        for i in range(self.delay_model.repeats):
            delayed_wave[i * n_delay_time_samples : i * n_delay_time_samples + len(wave)] += wave * (self.delay_model.feedback ** i)

        return delayed_wave[: len(wave)] if self.delay_model.do_trim else delayed_wave # Trim to original length
