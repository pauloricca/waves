import numpy as np
from config import SAMPLE_RATE

# Global constants and functions available in all expressions
GLOBAL_CONSTANTS = {
    # Math constants
    'pi': np.pi,
    'tau': 2 * np.pi,
    'e': np.e,
    
    # Sample rate
    'sr': SAMPLE_RATE,
    'sample_rate': SAMPLE_RATE,
    
    # NumPy functions (vectorized - work on both scalars and arrays)
    'sin': np.sin,
    'cos': np.cos,
    'tan': np.tan,
    'abs': np.abs,
    'clip': np.clip,
    'log': np.log,
    'log10': np.log10,
    'log2': np.log2,
    'exp': np.exp,
    'sqrt': np.sqrt,
    'pow': np.power,
    'max': np.maximum,
    'min': np.minimum,
    'floor': np.floor,
    'ceil': np.ceil,
    'round': np.round,
    'sign': np.sign,
    
    # Array creation
    'zeros': np.zeros,
    'ones': np.ones,
    'linspace': np.linspace,
    'arange': np.arange,
    
    # Useful aggregations
    'sum': np.sum,
    'mean': np.mean,
    'std': np.std,
    
    # NumPy itself for advanced usage
    'np': np,
}

# User-defined variables (loaded from YAML vars: section)
USER_VARIABLES = {}

def set_user_variables(vars_dict: dict):
    """Set user variables from YAML"""
    global USER_VARIABLES
    USER_VARIABLES = vars_dict.copy() if vars_dict else {}

def get_expression_context(render_params: dict, time: float, num_samples: int) -> dict:
    """
    Build the complete context for expression evaluation.
    Priority (highest priority overwrites lower):
    1. Global constants (lowest priority - base layer)
    2. User variables
    3. Render params  
    4. Special runtime variables (highest priority)
    
    Optimized: Start with global constants, layer on top to minimize dict operations.
    """
    # Start with global constants (we need these in almost all expressions)
    # This copy is necessary but happens only once per render
    context = GLOBAL_CONSTANTS.copy()
    
    # Layer on user variables (overwrite globals if there's a conflict)
    if USER_VARIABLES:
        context.update(USER_VARIABLES)
    
    # Layer on render params (overwrite user vars if there's a conflict)
    if render_params:
        context.update(render_params)
    
    # Add runtime-specific values (highest priority - overwrite everything)
    context['time'] = time
    context['t'] = time
    context['samples'] = num_samples
    context['n'] = num_samples
    
    return context

def evaluate_expression(expr_string: str, context: dict, num_samples: int = None):
    """
    Evaluate an expression string.
    Returns scalar or array depending on expression result.
    """
    try:
        compiled = compile(expr_string, '<expression>', 'eval')
        result = eval(compiled, {"__builtins__": {}}, context)
        
        # Convert result
        if isinstance(result, np.ndarray):
            return result
        elif isinstance(result, (int, float, np.number)):
            if num_samples is not None:
                return np.full(num_samples, float(result), dtype=np.float32)
            return float(result)
        else:
            # Let it through - might be used in further expressions
            return result
            
    except Exception as e:
        raise ValueError(f"Error evaluating expression '{expr_string}': {e}")
