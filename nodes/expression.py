from __future__ import annotations
import numpy as np
from typing import Union
from pydantic import ConfigDict
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from utils import empty_mono, empty_stereo, to_stereo, to_mono, is_stereo


class ExpressionNodeModel(BaseNodeModel):
    model_config = ConfigDict(extra='allow')  # Allow arbitrary named parameters
    exp: Union[str, int, float]  # The expression to evaluate (string expression or numeric constant)


class ExpressionNode(BaseNode):
    def __init__(self, model: ExpressionNodeModel, node_id: str, state=None, do_initialise_state=True):
        from expression_globals import compile_expression
        super().__init__(model, node_id, state, do_initialise_state)
        
        # ExpressionNode is stereo-capable - it can process and pass through stereo signals
        
        # Compile the main expression using centralized function
        try:
            self.compiled_exp, self.exp_value, self.is_constant = compile_expression(model.exp)
        except SyntaxError as e:
            # Add context about which node has the error
            # Clean up node_id by removing model class names for readability
            clean_id = node_id.replace('NodeModel', '').replace('.root.', '.')
            raise SyntaxError(
                f"\nExpression syntax error in '{clean_id}':\n{str(e)}"
            ) from e
        
        # Store all extra arguments (both nodes and raw values)
        # Pydantic with extra='allow' stores them in __pydantic_extra__
        self.args = {}
        self.compiled_args = {}  # Pre-compiled expressions (string or numeric)
        if hasattr(model, '__pydantic_extra__') and model.__pydantic_extra__:
            for field_name, field_value in model.__pydantic_extra__.items():
                if isinstance(field_value, BaseNodeModel):
                    self.args[field_name] = self.instantiate_child_node(field_value, field_name)
                else:
                    # String expression or numeric constant - compile it once
                    self.compiled_args[field_name] = compile_expression(field_value)
        
        # Track total samples rendered for duration checking
        if do_initialise_state:
            self.state.total_samples_rendered = 0
    

    def _do_render(self, num_samples=None, context=None, **params):
        from expression_globals import get_expression_context, evaluate_compiled
        
        # Resolve num_samples first (handles None case)
        num_samples = self.resolve_num_samples(num_samples)
        if num_samples is None:
            raise ValueError("Expression node requires explicit duration")
        
        # Track original num_samples for param truncation check
        original_num_samples = num_samples
        
        # Check if we've already rendered the full duration
        if self.duration is not None:
            from config import SAMPLE_RATE
            max_samples = int(self.duration * SAMPLE_RATE)
            
            # If we've already rendered everything, return empty
            if self.state.total_samples_rendered >= max_samples:
                return empty_mono()
            
            # If num_samples would exceed duration, limit it
            remaining_samples = max_samples - self.state.total_samples_rendered
            if remaining_samples <= 0:
                return empty_mono()
            num_samples = min(num_samples, remaining_samples)
        
        # Handle constant case - no need to evaluate anything
        if self.is_constant:
            return np.full(num_samples, self.exp_value, dtype=np.float32)
        
        # Build base context
        eval_context = get_expression_context(
            render_params=params,
            time=self.time_since_start,
            num_samples=num_samples,
            render_context=context
        )
        
        # If we limited num_samples due to our own duration, truncate param arrays NOW
        # (before rendering children, since params come from parent with original buffer size)
        if num_samples < original_num_samples:
            for name in list(eval_context.keys()):
                value = eval_context[name]
                if isinstance(value, np.ndarray) and len(value) > num_samples:
                    eval_context[name] = value[:num_samples]
        
        # Evaluate all arguments and add to context
        all_children_finished = True
        for name, value in self.args.items():
            if isinstance(value, BaseNode):
                # Render node (may return mono or stereo)
                wave = value.render(num_samples, context,
                                   **self.get_params_for_children(params))
                
                # Track if any child is still producing samples
                if len(wave) > 0:
                    all_children_finished = False
                
                # If child returned fewer samples, pad with zeros
                if len(wave) < num_samples:
                    if not is_stereo(wave):
                        # Mono - simple padding
                        wave = np.pad(wave, (0, num_samples - len(wave)), mode='constant', constant_values=0)
                    else:
                        # Stereo or multi-channel - pad only time axis (first dimension)
                        pad_width = [(0, num_samples - len(wave))] + [(0, 0)] * (wave.ndim - 1)
                        wave = np.pad(wave, pad_width, mode='constant', constant_values=0)
                
                # Expression node works with mono signals for mathematical operations
                # Convert stereo to mono for use in expressions
                if is_stereo(wave):
                    wave = to_mono(wave)
                
                eval_context[name] = wave
            # Note: scalar values are handled below in compiled_args
        
        # If all children have finished (returned empty arrays), signal completion
        if all_children_finished and len(self.args) > 0:
            return empty_mono()
        
        # Evaluate compiled arguments (expressions and constants)
        for name, compiled_info in self.compiled_args.items():
            result = evaluate_compiled(compiled_info, eval_context, num_samples)
            eval_context[name] = result
        
        # Evaluate main expression
        result = evaluate_compiled((self.compiled_exp, self.exp_value, self.is_constant), eval_context, num_samples)
        
        # Convert result to appropriate array
        output = None
        if isinstance(result, np.ndarray):
            if len(result) != num_samples:
                if len(result) == 1:
                    output = np.full(num_samples, result[0], dtype=np.float32)
                elif len(result) < num_samples:
                    # Pad with last value
                    padding = np.full(num_samples - len(result), result[-1])
                    output = np.concatenate([result, padding])
                else:
                    # Truncate
                    output = result[:num_samples]
            else:
                output = result.astype(np.float32)
        elif isinstance(result, (int, float, np.number)):
            output = np.full(num_samples, float(result), dtype=np.float32)
        else:
            raise ValueError(f"Expression returned unsupported type: {type(result)}")
        
        # Expression node always returns mono (mathematical operations work on mono signals)
        # Stereo child signals were converted to mono earlier for use in expressions
        
        # Track samples rendered
        self.state.total_samples_rendered += len(output)
        
        return output


EXPRESSION_DEFINITION = NodeDefinition(
    name="expression",
    model=ExpressionNodeModel,
    node=ExpressionNode
)
