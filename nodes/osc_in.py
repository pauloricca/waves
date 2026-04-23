from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from pydantic import ConfigDict, field_validator

from config import BUFFER_SIZE, OSC_INPUT_HOST, OSC_INPUT_PORT
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.node_utils.osc_utils import OSC_DEBUG, OscInputManager
from nodes.node_utils.range_mapper import RangeMapper
from nodes.wavable_value import WavableValue
from utils import get_last_or_default


class OscInModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    address: str
    arg_index: int = 0
    initial: float = 0.0
    range: Tuple[WavableValue, WavableValue] = (0.0, 1.0)
    host: str = OSC_INPUT_HOST
    port: int = OSC_INPUT_PORT
    duration: float = math.inf

    @field_validator('range', mode='before')
    @classmethod
    def validate_range(cls, v):
        if v is None:
            return (0.0, 1.0)

        if isinstance(v, (list, tuple)):
            if len(v) != 2:
                raise ValueError(f"range must have exactly 2 values, got {len(v)}")
            return tuple(v)

        raise ValueError("range must be a list or tuple with exactly 2 values")


class OscInNode(BaseNode):
    def __init__(self, model: OscInModel, node_id: str, state=None, do_initialise_state=True):
        super().__init__(model, node_id, state, do_initialise_state)
        self.address = model.address
        self.arg_index = max(0, int(model.arg_index))
        self.host = model.host
        self.port = model.port

        self.range_mapper = RangeMapper.from_model_range(
            self, model.range, "range",
            from_range=(0.0, 1.0),
        )

        if (
            model.range
            and isinstance(model.range[0], (int, float))
            and isinstance(model.range[1], (int, float))
        ):
            self.set_monitor_range(float(model.range[0]), float(model.range[1]))

        self.set_monitor_color_scheme('value')

        if do_initialise_state:
            self.state.current_normalized_value = float(model.initial)
            self.state.last_output_value = None

        self.osc_manager = OscInputManager()
        self.osc_manager.ensure_listener(self.host, self.port)

    def _process_osc_messages(self):
        new_value = self.osc_manager.get_value(
            self.address,
            self.host,
            self.port,
            self.arg_index,
        )
        if new_value is None:
            return

        self.state.current_normalized_value = float(new_value)
        if OSC_DEBUG:
            print(
                f"OSC {self.address}[{self.arg_index}] "
                f"on {self.host}:{self.port}: {self.state.current_normalized_value:.3f}"
            )

    def _do_render(self, num_samples=None, context=None, **params):
        if num_samples is None:
            num_samples = BUFFER_SIZE
            self._last_chunk_samples = num_samples

        self._process_osc_messages()

        normalized_wave = np.full(
            num_samples,
            self.state.current_normalized_value,
            dtype=np.float32,
        )

        if self.state.last_output_value is None:
            if self.range_mapper:
                initial_output = self.range_mapper.map(
                    np.array([self.state.current_normalized_value], dtype=np.float32),
                    1,
                    context,
                    **params,
                )
                self.state.last_output_value = float(initial_output[0])
            else:
                self.state.last_output_value = float(self.state.current_normalized_value)

        if self.range_mapper:
            target_wave = self.range_mapper.map(normalized_wave, num_samples, context, **params)
        else:
            target_wave = normalized_wave

        if num_samples == 1:
            output_wave = target_wave.astype(np.float32)
        else:
            interpolation_factor = np.linspace(0, 1, num_samples, dtype=np.float32)
            output_wave = (
                self.state.last_output_value * (1 - interpolation_factor)
                + target_wave * interpolation_factor
            )

        self.state.last_output_value = get_last_or_default(output_wave, self.state.last_output_value)
        return output_wave


OSC_IN_DEFINITION = NodeDefinition("osc_in", OscInNode, OscInModel)
