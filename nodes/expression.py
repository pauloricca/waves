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
    def __init__(self, model: ExpressionNodeModel, node_id: str, state=None, do_initialise_state=True):
        from expression_globals import compile_expression
        super().__init__(model, node_id, state, do_initialise_state)
        
        # Compile the main expression using centralized function
        self.compiled_exp, self.exp_value, self.is_constant = compile_expression(model.exp)
        
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
    

    def _do_render(self, num_samples=None, context=None, num_channels=1, **params):
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
                return np.array([], dtype=np.float32)
            
            # If num_samples would exceed duration, limit it
            remaining_samples = max_samples - self.state.total_samples_rendered
            if remaining_samples <= 0:
                return np.array([], dtype=np.float32)
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
            
            # Save child node outputs before recreating context
            child_outputs = {name: val for name, val in eval_context.items() 
                           if name in self.args}
            
            # Update base context arrays to match
            eval_context = get_expression_context(
                render_params=params,
                time=self.time_since_start,
                num_samples=num_samples,
                render_context=context
            )
            
            # Truncate ALL arrays in eval_context to match num_samples
            # This includes arrays from params (like 'v' from parent context)
            for name in list(eval_context.keys()):
                value = eval_context[name]
                if isinstance(value, np.ndarray) and len(value) > num_samples:
                    eval_context[name] = value[:num_samples]
            
            # Re-add child node outputs (already truncated if needed)
            for name, value in child_outputs.items():
                if isinstance(value, np.ndarray) and len(value) > num_samples:
                    eval_context[name] = value[:num_samples]
                else:
                    eval_context[name] = value
        
        # Evaluate compiled arguments (expressions and constants)
        for name, compiled_info in self.compiled_args.items():
            result = evaluate_compiled(compiled_info, eval_context, num_samples)
            eval_context[name] = result
            # DEBUG: Check if compiled arg created wrong-sized array
            if isinstance(result, np.ndarray) and len(result) != num_samples:
                print(f"DEBUG EXPR: Compiled arg '{name}' created array of length {len(result)}, expected {num_samples}")
        
        # DEBUG: Check all array sizes before evaluation
        arrays_ok = True
        for name, value in eval_context.items():
            if isinstance(value, np.ndarray):
                if len(value) != num_samples:
                    print(f"DEBUG EXPR: {name} has length {len(value)}, expected {num_samples}")
                    arrays_ok = False
        
        if not arrays_ok:
            print(f"DEBUG EXPR: Expression: {self.exp_value}")
            print(f"DEBUG EXPR: num_samples: {num_samples}")
            print(f"DEBUG EXPR: self.args keys: {list(self.args.keys())}")
            print(f"DEBUG EXPR: self.compiled_args keys: {list(self.compiled_args.keys())}")
        
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
        
        # Track samples rendered
        self.state.total_samples_rendered += len(output)
        
        return output


EXPRESSION_DEFINITION = NodeDefinition(
    name="expression",
    model=ExpressionNodeModel,
    node=ExpressionNode
)
