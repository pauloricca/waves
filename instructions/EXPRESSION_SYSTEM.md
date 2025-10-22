# Expression System Implementation

## Overview

A complete expression system has been implemented that allows dynamic evaluation of Python expressions in YAML node definitions. This enables flexible, mathematical sound design with variables, functions, and array operations.

## Features Implemented

### 1. Expression Node (`expression`)
A new node type that evaluates Python expressions with arbitrary named arguments.

**Parameters:**
- `exp` (str): The expression to evaluate
- `duration` (float, optional): Duration in seconds
- `id` (str, optional): Node identifier for referencing
- Any additional parameters become variables in the expression

**Example:**
```yaml
fm_sound:
  expression:
    exp: "carrier * (1 + mod * 0.3)"
    duration: 1
    carrier:
      osc:
        type: sin
        freq: 440
    mod:
      osc:
        type: sin
        freq: 220
```

### 2. Expression Strings in WavableValue
Any WavableValue parameter can now be a string expression.

**Example:**
```yaml
my_sound:
  osc:
    type: sin
    freq: "440 * 2"  # String = expression
    amp: "0.5"  # Can use expressions for any parameter
    duration: 1
```

### 3. Expression Strings in Scalar Parameters
Scalar parameters (like envelope times) can be expressions.

**Example:**
```yaml
envelope_sound:
  envelope:
    attack: "60 / bpm * 0.05"  # Calculated from BPM
    release: "60 / bpm * 0.3"
    signal:
      osc:
        type: sin
        freq: 440
```

### 4. User Variables
Define global variables in the YAML file using the `vars:` section.

**Example:**
```yaml
vars:
  bpm: 140
  root_note: 261.63
  attack_time: 0.02

my_sound:
  osc:
    freq: "root_note * 2"  # Uses user variable
    duration: "60 / bpm * 4"  # 4 beats
```

### 5. Global Constants and Functions
The following are available in all expressions:

**Math Constants:**
- `pi`, `tau`, `e`

**Sample Rate:**
- `sr`, `sample_rate`

**NumPy Functions (vectorized):**
- Trig: `sin`, `cos`, `tan`
- Basic: `abs`, `sqrt`, `exp`, `log`, `log10`, `log2`, `pow`
- Array ops: `clip`, `max`, `min`, `floor`, `ceil`, `round`, `sign`
- Creation: `zeros`, `ones`, `linspace`, `arange`
- Stats: `sum`, `mean`, `std`
- Direct access: `np` (full NumPy module)

**Runtime Variables:**
- `time` or `t`: Current time since node start
- `samples` or `n`: Number of samples being rendered
- Render params: `freq`/`f` (frequency), `amp`/`a` (amplitude), `gate` (>= 0.5 = sustaining), `duration`, and any custom params passed by parent nodes

## Architecture

### Files Added
1. **`nodes/expression_globals.py`**: Global constants, user variables, and evaluation functions
2. **`nodes/expression.py`**: Expression node implementation

### Files Modified
1. **`nodes/wavable_value.py`**: Added expression string support
2. **`nodes/node_utils/base_node.py`**: Added `eval_scalar()` helper method
3. **`nodes/node_utils/node_registry.py`**: Registered expression node
4. **`sound_library.py`**: Added user variable loading
5. **`nodes/envelope.py`**: Updated to accept expression strings for timing parameters

## Usage Examples

### Simple Math
```yaml
test:
  expression:
    exp: "sin(t * tau * 440) * 0.5"
    duration: 1
```

### Array Operations
```yaml
distortion:
  expression:
    exp: "clip(signal * 2, -1, 1)"
    duration: 1
    signal:
      osc:
        type: sin
        freq: 440
```

### FM Synthesis
```yaml
fm_bell:
  expression:
    exp: "carrier + carrier * mod * 0.5"
    duration: 2
    carrier:
      osc:
        type: sin
        freq: "root_note"
    mod:
      osc:
        type: sin
        freq: "root_note * 2"
```

### Wave Mixing
```yaml
mix:
  expression:
    exp: "(a + b) / 2"
    duration: 1
    a:
      osc:
        type: sin
        freq: 440
    b:
      osc:
        type: tri
        freq: 880
```

### Time-based Envelopes
```yaml
synth:
  envelope:
    attack: "60 / bpm * 0.05"
    decay: "60 / bpm * 0.1"
    sustain: 0.7
    release: "60 / bpm * 0.3"
    signal:
      osc:
        freq: "root_note * 2"
```

## Performance

- Expressions are compiled once at node initialization
- Uses Python's native `eval()` with restricted builtins for safety
- NumPy vectorization ensures array operations are fast
- No overhead for non-expression values
- All operations work in both realtime and non-realtime modes

## Design Principles

1. **Simplicity**: Any string is an expression (no special prefix needed)
2. **Dynamic**: All evaluation happens at render time (supports realtime parameter changes)
3. **Flexible**: Works with scalars, arrays, and nodes seamlessly
4. **Pythonic**: Uses familiar Python syntax and NumPy operations
5. **Safe**: Restricted eval context prevents dangerous operations

## Future Enhancements

Potential additions (not yet implemented):
- Musical note names (A4, C#5, etc.)
- Musical interval constants (semitone, fifth, octave as multipliers)
- Beat/time division constants (beat, bar, eighth, sixteenth)
- More DSP functions (filters, delays, etc.)
- Array slicing and indexing in expressions
- Conditional expressions
