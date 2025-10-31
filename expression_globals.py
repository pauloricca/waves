import math
import numpy as np
from config import SAMPLE_RATE
from random import uniform, choice

def rand(a: float | list, b: float | None = None) -> float:
    """Return a random value based on input pattern:
       - rand(a, b): uniform random between a and b
       - rand(a): uniform random between 0 and a
       - rand([list]): random choice from list"""
    # Handle different input patterns for rand function
    if isinstance(a, list):
        return choice(a)
    elif b is None:
        return uniform(0, a)
    # Regular case: uniform between a and b
    return uniform(a, b)

# Global constants and functions available in all expressions
GLOBAL_CONSTANTS = {
    # Math constants
    'pi': np.pi,
    'tau': 2 * np.pi,
    'e': np.e,
    'inf': math.inf,
    'infinite': math.inf,
    'infinity': math.inf,
    

    'rand': rand,
    
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
    
    # Useful aggregations
    'sum': np.sum,
    'mean': np.mean,
    'std': np.std,
    
    # NumPy itself for advanced usage
    'np': np,

    # Intervals (as frequency multipliers)
    'octave': 2.0,
    'whole': 2.0 ** (2.0 / 12.0),
    'semi': 2.0 ** (1.0 / 12.0),
    'whole_tone': 2.0 ** (2.0 / 12.0),
    'semitone': 2.0 ** (1.0 / 12.0),
    'fifth': 2.0 ** (7.0 / 12.0),
    'fourth': 2.0 ** (5.0 / 12.0),
    'major_third': 2.0 ** (4.0 / 12.0),
    'minor_third': 2.0 ** (3.0 / 12.0),
    'major_second': 2.0 ** (2.0 / 12.0),
    'minor_second': 2.0 ** (1.0 / 12.0),

    # Scales
    'chromatic': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'major': [0, 2, 4, 5, 7, 9, 11],
    'minor': [0, 2, 3, 5, 7, 8, 10],
    'major_pentatonic': [0, 2, 4, 7, 9],
    'minor_pentatonic': [0, 3, 5, 7, 10],
    'dorian': [0, 2, 3, 5, 7, 9, 10],
    'phrygian': [0, 1, 3, 5, 7, 8, 10],
    'lydian': [0, 2, 4, 6, 7, 9, 11],
    'mixolydian': [0, 2, 4, 5, 7, 9, 10],
    'aeolian': [0, 2, 3, 5, 7, 8, 10],
    'locrian': [0, 1, 3, 5, 6, 8, 10],
    'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
    'melodic_minor': [0, 2, 3, 5, 7, 9, 11],
    'double_harmonic': [0, 1, 4, 5, 7, 8, 11],
    'hungarian_minor': [0, 2, 3, 6, 7, 8, 11],
    'persian': [0, 1, 4, 5, 6, 8, 11],
    'enigmatic': [0, 1, 4, 6, 8, 10, 11],
    'neapolitan_minor': [0, 1, 3, 5, 7, 8, 11],
    'neapolitan_major': [0, 1, 3, 5, 7, 9, 11],
    'prometheus': [0, 2, 4, 6, 9, 10],
    'whole_tone': [0, 2, 4, 6, 8, 10],
    'blues': [0, 3, 5, 6, 7, 10],
    'minor_blues': [0, 3, 5, 6, 7, 10],
    'major_blues': [0, 2, 3, 4, 7, 9],
    'gypsy': [0, 2, 3, 6, 7, 8, 10],
    'japanese': [0, 1, 5, 7, 8],
    'hirajoshi': [0, 2, 3, 7, 8],
    'kumoi': [0, 2, 3, 7, 9],
    'yo': [0, 2, 5, 7, 9],
    'iwato': [0, 1, 5, 6, 10],
    'pelog': [0, 1, 3, 4, 7, 8],
    'augmented': [0, 3, 4, 7, 8, 11],
    'bebop_major': [0, 2, 4, 5, 7, 8, 9, 11],
    'bebop_minor': [0, 2, 3, 5, 7, 8, 9, 10],
}

# Musical note constants (C0 to G10)
def _generate_note_constants():
    notes = {}
    note_names = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b']
    # MIDI note 12 = C0, MIDI note 127 = G9, but let's go up to G10 for completeness
    for midi in range(12, 128 + 12):  # 12 to 139 (G10)
        octave = (midi // 12) - 1
        name = note_names[midi % 12] + str(octave)
        freq = 440.0 * 2 ** ((midi - 69) / 12)
        notes[name] = freq
        # Also add uppercase variant for convenience
        notes[name.upper()] = freq
    return notes

GLOBAL_CONSTANTS.update(_generate_note_constants())

# User-defined variables (loaded from YAML vars: section)
USER_VARIABLES = {}

def set_user_variables(vars_dict: dict):
    """Set user variables from YAML"""
    global USER_VARIABLES
    USER_VARIABLES = vars_dict.copy() if vars_dict else {}

def get_expression_context(render_params: dict, time: float, num_samples: int, render_context=None) -> dict:
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
    
    # Add render context if provided (for advanced usage like node() function)
    if render_context is not None:
        context['context'] = render_context
    
    return context

def compile_expression(expr):
    """
    Compile an expression for evaluation.
    
    Args:
        expr: Can be a string (expression to compile), numeric value (returns as-is),
              or already compiled code object (returns as-is)
    
    Returns:
        Tuple of (compiled_code_or_None, numeric_value_or_None, is_constant)
        - If numeric: (None, value, True)
        - If expression: (compiled, None, False)
    """
    if isinstance(expr, (int, float)):
        # Numeric constant - no compilation needed
        return (None, float(expr), True)
    elif isinstance(expr, str):
        # String expression - compile it
        return (compile(expr, '<expression>', 'eval'), None, False)
    elif hasattr(expr, 'co_code'):
        # Already compiled
        return (expr, None, False)
    else:
        raise ValueError(f"Cannot compile expression of type {type(expr)}")


def evaluate_compiled(compiled_info, context: dict, num_samples: int = None):
    """
    Evaluate a pre-compiled expression info tuple.
    
    Args:
        compiled_info: Tuple from compile_expression (compiled_code, const_value, is_constant)
        context: Dictionary of variables available during evaluation
        num_samples: If provided, convert scalar results to arrays of this length
    
    Returns:
        Scalar or array depending on expression result and num_samples.
    """
    compiled, const_value, is_constant = compiled_info
    
    # If it's a constant, return it directly
    if is_constant:
        if num_samples is not None:
            return np.full(num_samples, const_value, dtype=np.float32)
        return const_value
    
    # Evaluate the compiled expression
    result = eval(compiled, {"__builtins__": {}}, context)
    
    # Convert result
    if isinstance(result, np.ndarray):
        # Handle boolean arrays by converting to float
        if result.dtype == bool or result.dtype == np.bool_:
            return result.astype(np.float32)
        return result
    elif isinstance(result, (bool, np.bool_)):
        # Convert boolean to float (True->1.0, False->0.0)
        value = 1.0 if result else 0.0
        if num_samples is not None:
            return np.full(num_samples, value, dtype=np.float32)
        return value
    elif isinstance(result, (int, float, np.number)):
        if num_samples is not None:
            return np.full(num_samples, float(result), dtype=np.float32)
        return float(result)
    else:
        # Let it through - might be used in further expressions
        return result


def evaluate_expression(expr, context: dict, num_samples: int = None):
    """
    Evaluate an expression (compiles if needed).
    
    Args:
        expr: Can be a string, numeric value, or compiled code object
        context: Dictionary of variables available during evaluation
        num_samples: If provided, convert scalar results to arrays of this length
    
    Returns:
        Scalar or array depending on expression result and num_samples.
    """
    try:
        # Compile and evaluate
        compiled_info = compile_expression(expr)
        return evaluate_compiled(compiled_info, context, num_samples)
            
    except Exception as e:
        expr_str = expr if isinstance(expr, str) else repr(expr)
        raise ValueError(f"Error evaluating expression '{expr_str}': {e}")
