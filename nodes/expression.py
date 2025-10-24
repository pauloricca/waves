from __future__ import annotations
import numpy as np
from typing import Union
from pydantic import ConfigDict
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition


class ExpressionNodeModel(BaseNodeModel):
    model_config = ConfigDict(extra='allow')  # Allow arbitrary named parameters
    exp: Union[str, int, float]  # The expression to evaluate (string expression or numeric constant)


class ExpressionNode(BaseNode):
    def __init__(self, model: ExpressionNodeModel, state=None, hot_reload=False):
        from nodes.node_utils.instantiate_node import instantiate_node
        from expression_globals import compile_expression
        super().__init__(model, state, hot_reload)
        
        # Compile the main expression using centralized function
        self.compiled_exp, self.exp_value, self.is_constant = compile_expression(model.exp)
        
        # Store all extra arguments (both nodes and raw values)
        # Pydantic with extra='allow' stores them in __pydantic_extra__
        self.args = {}
        self.compiled_args = {}  # Pre-compiled expressions (string or numeric)
        if hasattr(model, '__pydantic_extra__') and model.__pydantic_extra__:
            for field_name, field_value in model.__pydantic_extra__.items():
                if isinstance(field_value, BaseNodeModel):
                    self.args[field_name] = instantiate_node(field_value, hot_reload=hot_reload)
                else:
                    # String expression or numeric constant - compile it once
                    self.compiled_args[field_name] = compile_expression(field_value)
    
    def _do_render(self, num_samples=None, context=None, **params):
        from expression_globals import get_expression_context, evaluate_compiled
        
        num_samples = self.resolve_num_samples(num_samples)
        if num_samples is None:
            raise ValueError("Expression node requires explicit duration")
        
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
        
        # Track the actual minimum number of samples from all child nodes
        actual_num_samples = num_samples
        
        # Evaluate all arguments and add to context
        for name, value in self.args.items():
            if isinstance(value, BaseNode):
                # Render node
                wave = value.render(num_samples, context, 
                                   **self.get_params_for_children(params))
                eval_context[name] = wave
                # Track if child returned fewer samples
                if len(wave) < actual_num_samples:
                    actual_num_samples = len(wave)
            # Note: scalar values are handled below in compiled_args
        
        # If any child returned fewer samples, we need to handle completion
        if actual_num_samples < num_samples:
            # If any child returned 0 samples, signal completion
            if actual_num_samples == 0:
                return np.array([], dtype=np.float32)
            
            # Otherwise, truncate all arrays to match the shortest child
            num_samples = actual_num_samples
            # Update base context arrays to match
            eval_context = get_expression_context(
                render_params=params,
                time=self.time_since_start,
                num_samples=num_samples,
                render_context=context
            )
            # Truncate all child node outputs
            for name, value in eval_context.items():
                if isinstance(value, np.ndarray) and len(value) > num_samples:
                    eval_context[name] = value[:num_samples]
        
        # Evaluate compiled arguments (expressions and constants)
        for name, compiled_info in self.compiled_args.items():
            result = evaluate_compiled(compiled_info, eval_context, num_samples)
            eval_context[name] = result
        
        # Evaluate main expression
        result = evaluate_compiled((self.compiled_exp, self.exp_value, self.is_constant), eval_context, num_samples)
        
        # Convert result to appropriate array
        if isinstance(result, np.ndarray):
            if len(result) != num_samples:
                if len(result) == 1:
                    return np.full(num_samples, result[0], dtype=np.float32)
                elif len(result) < num_samples:
                    # Pad with last value
                    padding = np.full(num_samples - len(result), result[-1])
                    return np.concatenate([result, padding])
                else:
                    # Truncate
                    return result[:num_samples]
            return result.astype(np.float32)
        elif isinstance(result, (int, float, np.number)):
            return np.full(num_samples, float(result), dtype=np.float32)
        else:
            raise ValueError(f"Expression returned unsupported type: {type(result)}")


EXPRESSION_DEFINITION = NodeDefinition(
    name="expression",
    model=ExpressionNodeModel,
    node=ExpressionNode
)
