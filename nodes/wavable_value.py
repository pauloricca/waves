    
from __future__ import annotations
from enum import Enum
from typing import List, Union
import numpy as np
from pydantic import ConfigDict
from config import SAMPLE_RATE
from utils import time_to_samples, ensure_array
from constants import RenderArgs
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from scipy.interpolate import PchipInterpolator


class InterpolationTypes(str, Enum):
    LINEAR = "LINEAR"
    SMOOTH = "SMOOTH"
    STEP = "STEP"


# Allow strings (expressions) within interpolation lists
WavableValue = Union[float, int, List[Union[float, str, List[Union[float, str]]]], BaseNodeModel, str]
WavableValueNotModel = Union[float, int, List[Union[float, str, List[Union[float, str]]]], str]

class WavableValueModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    value: WavableValueNotModel
    interpolation: InterpolationTypes = InterpolationTypes.LINEAR


class WavableValueNode(BaseNode):
    def __init__(self, model: WavableValueModel, node_id: str, state=None, do_initialise_state=True):
        from expression_globals import compile_expression
        super().__init__(model, node_id, state, do_initialise_state)
        self.value: WavableValueNotModel = model.value
        self.interpolation_type = model.interpolation
        self.interpolated_values = None
        
        # Determine value type
        if isinstance(model.value, str):
            # String = expression - compile it and store the info
            self.compiled_info = compile_expression(model.value)
            self.value_type = 'expression'
        elif isinstance(model.value, (float, int)):
            self.value_type = 'scalar'
        elif isinstance(model.value, list):
            self.value_type = 'interpolated'
            # Pre-compile expressions in interpolation lists
            self._compile_interpolation_expressions()
        else:
            self.value_type = 'unknown'
    
    def _compile_interpolation_expressions(self):
        """Pre-compile any expression strings found in interpolation lists."""
        from expression_globals import compile_expression
        self.compiled_interpolation = []
        for item in self.value:
            if isinstance(item, str):
                # Single expression
                self.compiled_interpolation.append(compile_expression(item))
            elif isinstance(item, list):
                # [value, position] - compile value if it's a string
                if isinstance(item[0], str):
                    compiled_value = compile_expression(item[0])
                    # Keep position as-is (it should be numeric)
                    self.compiled_interpolation.append([compiled_value, item[1]])
                else:
                    # Already numeric
                    self.compiled_interpolation.append(item)
            else:
                # Already numeric
                self.compiled_interpolation.append(item)

    def _do_render(self, num_samples=None, context=None, **params):
        from expression_globals import get_expression_context, evaluate_compiled
        
        if self.value_type == 'expression':
            # Expression evaluation using centralized function
            if num_samples is None:
                duration = params.get(RenderArgs.DURATION, 0)
                if duration == 0:
                    raise ValueError("Duration required for expression")
                num_samples = time_to_samples(duration )
                self._last_chunk_samples = num_samples
            
            eval_context = get_expression_context(params, self.time_since_start, num_samples, context)
            result = evaluate_compiled(self.compiled_info, eval_context, num_samples)
            
            # Result is already properly formatted by evaluate_compiled
            return ensure_array(result, num_samples)
        
        elif self.value_type == 'scalar':
            # Scalar values
            if num_samples is None:
                # For scalar values, we need a duration to know how many samples to generate
                duration = params.get(RenderArgs.DURATION, 0)
                if duration == 0:
                    raise ValueError("Duration must be set for scalar wavable values when rendering full signal.")
                num_samples = time_to_samples(duration )
                self._last_chunk_samples = num_samples
            return np.full(num_samples, self.value, dtype=np.float32)
        
        elif self.value_type == 'interpolated':
            # Interpolated values
            duration = params.get(RenderArgs.DURATION, 0)

            if duration == 0:
                raise ValueError("Duration must be set somewhere above interpolated values.")
            
            if self.interpolated_values is None or len(self.interpolated_values) != time_to_samples(duration ):
                # Evaluate any expressions in the interpolation list first
                from expression_globals import get_expression_context, evaluate_compiled
                eval_context = get_expression_context(params, self.time_since_start, 1, context)
                evaluated_values = self._evaluate_interpolation_list(eval_context)
                self.interpolated_values = interpolate_values(evaluated_values, time_to_samples(duration ), self.interpolation_type)

            if num_samples is None:
                # Return the entire interpolated values
                self._last_chunk_samples = len(self.interpolated_values)
                return self.interpolated_values.copy()

            interpolated_values_section = self.interpolated_values[self.number_of_chunks_rendered: self.number_of_chunks_rendered + num_samples]

            if len(interpolated_values_section) < num_samples:
                padding = np.full(num_samples - len(interpolated_values_section), self.interpolated_values[-1])
                interpolated_values_section = np.concatenate([interpolated_values_section, padding])

            return interpolated_values_section.copy()
    
    def _evaluate_interpolation_list(self, eval_context):
        """Evaluate any compiled expressions in the interpolation list."""
        from expression_globals import evaluate_compiled
        
        evaluated = []
        for item in self.compiled_interpolation:
            if isinstance(item, tuple):
                # It's a compiled expression tuple (compiled_code, const_value, is_constant)
                value = evaluate_compiled(item, eval_context, num_samples=None)
                # Ensure it's a simple float
                if isinstance(value, np.ndarray):
                    value = float(value.flat[0]) if value.size > 0 else 0.0
                elif isinstance(value, (int, float)):
                    value = float(value)
                evaluated.append(value)
            elif isinstance(item, list):
                # [value/compiled, position]
                if isinstance(item[0], tuple):
                    # It's a compiled expression
                    value = evaluate_compiled(item[0], eval_context, num_samples=None)
                    # Ensure it's a simple float
                    if isinstance(value, np.ndarray):
                        value = float(value.flat[0]) if value.size > 0 else 0.0
                    elif isinstance(value, (int, float)):
                        value = float(value)
                    evaluated.append([value, item[1]])
                else:
                    # Already numeric
                    evaluated.append(item)
            else:
                # Already numeric - ensure it's float
                if isinstance(item, (int, float)):
                    evaluated.append(float(item))
                else:
                    evaluated.append(item)
        
        return evaluated



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