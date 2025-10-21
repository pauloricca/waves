from __future__ import annotations
import numpy as np
from pydantic import ConfigDict
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition


class ExpressionNodeModel(BaseNodeModel):
    model_config = ConfigDict(extra='allow')  # Allow arbitrary named parameters
    exp: str  # The expression to evaluate


class ExpressionNode(BaseNode):
    def __init__(self, model: ExpressionNodeModel):
        from nodes.node_utils.instantiate_node import instantiate_node
        super().__init__(model)
        self.exp_string = model.exp
        
        # Compile once for performance
        self.compiled_exp = compile(self.exp_string, '<expression>', 'eval')
        
        # Store all extra arguments (both nodes and raw values)
        # Pydantic with extra='allow' stores them in __pydantic_extra__
        self.args = {}
        self.compiled_string_args = {}  # Pre-compile string expressions
        if hasattr(model, '__pydantic_extra__') and model.__pydantic_extra__:
            for field_name, field_value in model.__pydantic_extra__.items():
                if isinstance(field_value, BaseNodeModel):
                    self.args[field_name] = instantiate_node(field_value)
                elif isinstance(field_value, str):
                    # String (expression) - compile it once
                    self.compiled_string_args[field_name] = compile(field_value, '<expression>', 'eval')
                elif isinstance(field_value, (int, float)):
                    # Scalar - store as-is
                    self.args[field_name] = field_value
    
    def _do_render(self, num_samples=None, context=None, **params):
        from expression_globals import get_expression_context
        
        num_samples = self.resolve_num_samples(num_samples)
        if num_samples is None:
            raise ValueError("Expression node requires explicit duration")
        
        # Build base context
        eval_context = get_expression_context(
            render_params=params,
            time=self.time_since_start,
            num_samples=num_samples
        )
        
        # Evaluate all arguments and add to context
        for name, value in self.args.items():
            if isinstance(value, BaseNode):
                # Render node
                wave = value.render(num_samples, context, 
                                   **self.get_params_for_children(params))
                eval_context[name] = wave
            else:
                # Scalar (int/float)
                eval_context[name] = value
        
        # Evaluate compiled string arguments
        for name, compiled_expr in self.compiled_string_args.items():
            result = eval(compiled_expr, {"__builtins__": {}}, eval_context)
            if isinstance(result, np.ndarray):
                eval_context[name] = result
            elif isinstance(result, (int, float, np.number)):
                eval_context[name] = np.full(num_samples, float(result), dtype=np.float32)
            else:
                eval_context[name] = result
        
        # Evaluate main expression
        result = eval(self.compiled_exp, {"__builtins__": {}}, eval_context)
        
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
