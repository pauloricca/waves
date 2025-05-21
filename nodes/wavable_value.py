    
from __future__ import annotations
from enum import Enum
from typing import List, Union
import numpy as np
from pydantic import ConfigDict
from config import SAMPLE_RATE
from constants import RenderArgs
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from scipy.interpolate import PchipInterpolator


class InterpolationTypes(str, Enum):
    LINEAR = "LINEAR"
    SMOOTH = "SMOOTH"
    STEP = "STEP"


WavableValue = Union[float, List[Union[float, List[float]]], BaseNodeModel]

class WavableValueModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    value: WavableValue
    interpolation: InterpolationTypes = InterpolationTypes.LINEAR


class WavableValueNode(BaseNode):
    def __init__(self, model: WavableValueModel):
        from nodes.node_utils.instantiate_node import instantiate_node
        super().__init__(model)
        self.value = model.value
        self.interpolation_type = model.interpolation
        self.wave_node = instantiate_node(model.value) if isinstance(model.value, BaseNodeModel) else None
        self.interpolated_values = None

    def render(self, num_samples, **kwargs):
        from nodes.oscillator import OSCILLATOR_RENDER_ARGS
        super().render(num_samples)

        duration = kwargs.get(RenderArgs.DURATION, 0)

        if self.wave_node:
            return self.wave_node.render(num_samples, **self.get_kwargs_for_children(kwargs, OSCILLATOR_RENDER_ARGS))
        elif isinstance(self.value, (float, int)):
            return np.array([self.value])
        if isinstance(self.value, list):
            if self.interpolated_values is None or len(self.interpolated_values) != int(duration * SAMPLE_RATE):
                self.interpolated_values = interpolate_values(self.value, int(duration * SAMPLE_RATE), self.interpolation_type)
            
            interpolated_values_section = self.interpolated_values[int(self.time_since_start * SAMPLE_RATE): int(self.time_since_start * SAMPLE_RATE + num_samples)]
            
            if len(interpolated_values_section) < num_samples:
                padding = np.full(num_samples - len(interpolated_values_section), self.interpolated_values[-1])
                interpolated_values_section = np.concatenate([interpolated_values_section, padding])

            return interpolated_values_section


def wavable_value_node_factory(value: WavableValue, interpolation: InterpolationTypes = InterpolationTypes.LINEAR):
    return WavableValueNode(WavableValueModel(value=value, interpolation=interpolation))


# Interpolates a list of values or a list of lists with relative positions
def interpolate_values(values, num_samples, interpolation_type: InterpolationTypes):
    # If all the values apart from the first and last are lists, we assume they are relative positions
    # It's ok for the first and last not to be lists, as we can assume 0 and 1 positions
    if all(isinstance(v, list) and len(v) == 2 for v in values[1:-1]):        
        # Assume 0 and 1 positions for the first and last values
        if not isinstance(values[0], list):
            values[0] = [values[0], 0]
        if not isinstance(values[-1], list):
            values[-1] = [values[-1], 1]

        # If the first value is not in position 0, we add an extra value at the beginning
        if values[0][1] != 0:
            values = [[values[0][0], 0]] + values
        # If the last value is not in position 1, we add an extra value at the end
        if values[-1][1] != 1:
            values = values + [[values[-1][0], 1]]

        # Handle list of lists with relative positions
        positions = [v[1] for v in values]
        values = [v[0] for v in values]
        if interpolation_type == InterpolationTypes.STEP.value:
            # Step interpolation
            positions = np.array(positions)
            values = np.array(values)
            interpolated_values = np.zeros(num_samples)
            for i in range(len(positions) - 1):
                start = int(positions[i] * num_samples)
                end = int(positions[i + 1] * num_samples)
                interpolated_values[start:end] = values[i]
        elif interpolation_type == InterpolationTypes.SMOOTH.value:
            # Smooth interpolation with a tighter curve
            positions = np.array(positions)
            values = np.array(values)
            x = np.linspace(0, 1, num_samples)
            interpolated_values = np.interp(x, positions, values)
            pchip = PchipInterpolator(positions, values)
            interpolated_values = pchip(x)
        else:
            # Linear interpolation
            positions = np.array(positions)
            values = np.array(values)
            interpolated_values = np.zeros(num_samples)
            for i in range(len(positions) - 1):
                start = int(positions[i] * num_samples)
                end = int(positions[i + 1] * num_samples)
                x = np.linspace(0, 1, end - start)
                interpolated_values[start:end] = np.interp(x, [0, 1], [values[i], values[i + 1]])
    elif len(values) > 1:
        # Handle simple list of values
        if interpolation_type == InterpolationTypes.STEP.value:
            # Step interpolation
            interpolated_values = np.repeat(values, num_samples // len(values))
        elif interpolation_type == InterpolationTypes.SMOOTH.value:
            # Smooth interpolation with a tighter curve
            positions = np.linspace(0, 1, len(values))
            pchip = PchipInterpolator(positions, values)
            x = np.linspace(0, 1, num_samples)
            interpolated_values = pchip(x)
        else:
            # Linear interpolation
            x = np.linspace(0, 1, len(values))
            x_interp = np.linspace(0, 1, num_samples)
            interpolated_values = np.interp(x_interp, x, values)
    else:
        interpolated_values = values[0]
    return interpolated_values