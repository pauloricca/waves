# GitHub Copilot Instructions

## Project Overview

This project is a python and numpy test bed for experimental sound design, to help me create sounds, sequences or compositions from scratch to be used in music production in flexible, low-level hands-on ways. The way sounds are created is through a kind of modular synthesis, where different modules (called nodes here) can be connected together procedurally to create complex sounds. The modules and connections are described in YAML format in the waves.yaml file.

There are different types of nodes, such as oscillators, filters, and effects, each with their own parameters and often parameters can be either set to scalar values or connected to the output of other nodes.

## Key Concepts

### Nodes

Nodes are objects that inherit from BaseNode (nodes/node_utils/base_node.py) and they can be instantiated by calling the instantiate_node function (nodes/node_utils/instantiate_node.py) using a definition from the yaml file, which gets translated into an object that inherits from BaseNodeModel (nodes/node_utils/base_node.py).

In short, a Node instance is defined in yaml, matching the signature of a NodeModel and is instantiated as a Node object at runtime.

A Node has a render method which takes in a number of samples to render and a set of parameters, and returns a numpy array of samples. The render method can use the parameters and the internal state of the node to generate what we call a wave. 

#### The render method

The render method is the main public interface for generating audio. It handles cross-cutting concerns like caching, recursion tracking, and timing, then delegates to the `_do_render()` method for the actual rendering logic.

**Architecture:**
- `render(num_samples, context, **params)` - Public method in BaseNode that:
  1. Creates a RenderContext if none provided (backwards compatibility)
  2. Handles caching for nodes with an `id` field
  3. Tracks recursion depth to prevent infinite feedback loops
  4. Updates timing information
  5. Delegates to `_do_render()` for actual rendering
  
- `_do_render(num_samples, context, **params)` - Protected method that subclasses override:
  - Contains the pure rendering logic specific to each node type
  - Should not handle caching, recursion, or timing (BaseNode handles that)
  - Must pass `context` to all child `render()` calls
  - Pass `context` as second positional parameter: `child.render(num_samples, context, **params)`

The render method is called multiple times (potentially in real time) to generate the final output wave. Each time it is called, it generates a chunk of samples (the size of which is defined by the num_samples parameter). We should take this into consideration when writing its implementation, we should take measures to ensure that this method is efficient and does not introduce artifacts, clicks or cracks in the output wave, for example as a result of value jumps between one chunk and the next. We might need to keep some internal state (or sometimes a buffer) in the node to ensure continuity between chunks.

The number of samples is optional. If not provided, the node should render the whole wave at once. This is useful for non-realtime rendering, where we want to generate the whole sound in one go. When the render function returns an empty array, it signals to the parent node that it has finished rendering and there is no more sound to be generated. This is useful for nodes that have a defined duration, such as an envelope or a sample player.

The render method receives a params dictionary, which contains the parameters that are passed on by parent nodes and can be used by the node to generate the wave. In order to keep these params organised, we use the RenderArgs enum (constants.py) to define the keys that can be used in the params dictionary. This way, we can avoid hardcoding strings in the code and make it easier to understand what each parameter is for.

### RenderContext

The RenderContext (nodes/node_utils/render_context.py) manages shared state during rendering:
- **Node output caching**: Stores outputs of nodes with an `id` field so they can be referenced by other nodes
- **Recursion tracking**: Prevents infinite loops by tracking recursion depth per node ID
- **Feedback loop control**: When recursion depth exceeds `MAX_RECURSION_DEPTH` (config.py), returns zeros to break the loop
- **Realtime chunk management**: In realtime mode, the context persists across chunks but clears cached outputs after each chunk

The context is passed through all render calls and enables advanced routing patterns like fan-out (one LFO modulating multiple oscillators) and feedback loops.

### Waves

Waves can be used as audio or as input for parameters of other nodes. In this sense, there is no destinction between audio and control (like in modular synths), they are all just waves. The objective is to create a unified representation of all types of signals and make the system very flexible.

### WavableValues

Some parameters of nodes can be fixed scalar values or waves – dynamic values that change over time. We call this a WavableValue (nodes/wavable_value.py). Waves are typically the output of a Node's render function but they can also be:

1. **Interpolated values**: A list of values to be used for interpolation. Each value can be a scalar or a list of two values, the first one is the value and the second one is the time it takes to get to that value, proportionally to the total time of the envelope. For example:
   `freq: [10000, [500, 0.01], [60, 0.02]]` means that the frequency starts at 10000 Hz, then goes to 500 Hz in 1% of the total time of the envelope, then goes to 60 Hz in 2% of the total time of the envelope, and then stays at 60 Hz for the rest of the time. The first and last values can have their positions omitted (assumed to be 0 and 1 respectively).

2. **Expression strings**: Any string value is treated as a Python expression that is evaluated at render time. Expressions have access to global constants, user-defined variables, render parameters, and NumPy functions. For example:
   `freq: "440 * 2"` evaluates to 880
   `freq: "root_note * 2"` uses a user variable

### Expression System

The expression system (nodes/expression_globals.py, nodes/expression.py) allows dynamic evaluation of Python expressions in YAML node definitions. This enables flexible, mathematical sound design with variables, functions, and array operations.

**Key features:**
- Any string value in YAML is treated as an expression (no special prefix needed)
- All evaluation happens at render time (fully dynamic, supports realtime parameter changes)
- Expressions are compiled once at node initialization for performance
- Works with both scalar values and NumPy arrays seamlessly
- NumPy operations are vectorized and work on entire audio buffers

**Available in expressions:**
- **Global constants**: `pi`, `tau`, `e`, `sr` (sample rate), `sample_rate`
- **Runtime variables**: `time` or `t` (time since start), `samples` or `n` (number of samples in chunk)
- **Render parameters**: `freq`/`f` (frequency), `amp`/`a` (amplitude), `gate` (>= 0.5 = sustaining), `duration`, and any custom params passed by parent nodes
- **User variables**: Defined in `vars:` section at the top of YAML file
- **NumPy functions**: `sin`, `cos`, `tan`, `abs`, `sqrt`, `exp`, `log`, `clip`, `max`, `min`, `floor`, `ceil`, `round`, `sign`, `zeros`, `ones`, `linspace`, `arange`, `sum`, `mean`, `std`, and full `np` module

**Expression node:**
The `expression` node type allows arbitrary multi-input operations with named arguments:
```yaml
my_sound:
  expression:
    exp: "carrier * (1 + mod * 0.3)"  # FM synthesis
    duration: 1
    carrier:  # Named argument becomes variable in expression
      osc:
        type: sin
        freq: 440
    mod:  # Another named argument
      osc:
        type: sin
        freq: 220
```

**Expression examples:**
- Scalar: `attack: "60 / bpm * 0.05"` (uses user variable)
- WavableValue: `freq: "440 * 2"` (simple math)
- With render params: `freq: "root_note * 2"`
- Array operations: `exp: "clip(signal * 2, -1, 1)"` (distortion)
- Time-based: `exp: "sin(t * tau * 440) * 0.5"` (synthesize from scratch)

**User variables:**
Define global variables at the top of waves.yaml:
```yaml
vars:
  bpm: 140
  root_note: 261.63
  attack_time: 0.02
```

These variables are then available in all expressions throughout the file.

**Implementation notes:**
- Expressions use Python's `eval()` with restricted builtins (`{"__builtins__": {}}`) for safety
- For scalar parameters that support expressions, use `self.eval_scalar(value, context, **params)` in node code
- WavableValue automatically handles expression strings when the value type is detected as 'expression'
- Expression node uses `ConfigDict(extra='allow')` to accept arbitrary named parameters, stored in `__pydantic_extra__`

## The YAML file

The yaml file (waves.yaml) is the main definitions file for the project. Each root level key identifies a different sound or sequence to be generated and the subsequent child keys define nodes and their parameters in a tree structure:

```yaml
sound_identifier:
  node_name_1:
    parameter_1: value_1
    parameter_2:
        node_name_2:
          parameter_a: value_a
          parameter_b: value_b
  node_name_3:
    (...)
```

The structure of the yaml file as is, at the moment, always with the sound identifiers as root notes, then a root node name, then its parameters, some of which can be other nodes, and so on recursively. In the future we might think of adding some syntax sugar and relax this rule.

### Node References and Advanced Routing

Nodes can now be given an `id` and referenced by other nodes, enabling flexible routing beyond simple tree structures:

**Defining a reusable node:**
```yaml
my_sound:
  osc:
    id: my_lfo  # Give this node an ID
    type: sin
    freq: 2
    range: [0, 1]
```

**Referencing a node:**
```yaml
my_sound:
  mix:
    signals:
      # Define an LFO with an id
      - osc:
          id: shared_lfo
          type: sin
          freq: 2
          range: [200, 800]
      
      # Use it to modulate frequency of multiple oscillators
      - osc:
          type: sin
          freq:
            reference:
              ref: shared_lfo
      
      - osc:
          type: tri
          freq:
            reference:
              ref: shared_lfo
```

**Feedback loops:**
Nodes can reference themselves or create circular dependencies. The system automatically detects and controls feedback loops using recursion depth tracking:

```yaml
feedback_delay:
  delay:
    id: feedback
    time: 0.3
    feedback: 0.5
    signal:
      reference:
        ref: feedback  # References itself - creates feedback loop
```

When recursion depth exceeds `MAX_RECURSION_DEPTH` (default 10), the system returns zeros to break the loop.

### Sub-Patching

You can reference root-level sounds as if they were node types, allowing sound reuse with parameter overrides:

```yaml
# Define base sounds
hihat:
  context:
    v: 1
    signal:
      # ... hihat definition

kick:
  mix:
    # ... kick definition

# Use them as sub-patches with custom parameters
my_composition:
  mix:
    signals:
      - hihat:
          v: 2       # Override v parameter
      - kick:
          amp: 0.5   # Override amp parameter
      - hihat:
          v: 0.3     # Another instance with different v
```

When a node type isn't recognized in the NODE_REGISTRY, the parser checks if it matches a root-level sound name. If found, it instantiates that sound and applies any provided parameters directly to its model. This enables modular composition and sound reuse.

## Running the code

To run the code we use the main waves.py file, followed by one parameter, which should match one of the root level keys in the yaml file. For example:

```bash
python waves.py sound_identifier
```

The program will then play the sound and stay idle until it detects a change in the yaml file, at which point it will re-render the sound and play it again, so that we can make changes and hear the results in real time.

There is a command-line based audio visualizer that shows the waveform of the sound being played (both in realtime and non-realtime modes).

## Creating a new type of Node

To create a new type of node, we create a new file with the node name in the nodes/ directory. The file should contain:
- a class that inherits from BaseNodeModel
- a class that inherits from BaseNode
- constant of type NodeDefinition called {NODE_NAME}_DEFINITION that attaches a node name, the model and the node class.

Then we need to add the new node to the NODE_REGISTRY in nodes/node_utils/node_registry.py.

**Important implementation details:**
- Override `_do_render(self, num_samples=None, context=None, **params)` not `render()`
- The `render()` method in BaseNode handles caching, recursion, and timing automatically
- Always pass `context` as the second positional parameter when calling child `render()` methods
- Ensure the node works correctly in both realtime and non-realtime modes
- In realtime mode, the node renders in chunks; in non-realtime, it may render the entire signal at once
- When `num_samples` is None, render the full signal (or raise an error if the node needs explicit duration)

## Configuration

In config.py we can set some global parameters for the project, such as the sample rate, the buffer size, etc. but also some options on the mode of operating like whether we play the sound in realtime (render in chunks) or pre-render the whole sound and play it back (non-realtime).

## Past and Future Work

In the earlier stages of the project, we only supported non-realtime playback and then slowly we started converting nodes to support realtime rendering. The goal is to have as many nodes supporting both realtime and non-realtime modes as possible. The nodes that don't support realtime rendering are in the nodes/non_realtime directory.

The objective of having realtime playback is to be able to use a MIDI controller or other input device and play the sounds live, as well as being able to tweak parameters in real time and hear the results immediately. Currently we do this with the midi and midi_cc nodes.

### Ideas for future work

When adding new features we should consider the following ideas for future work when making architectural decisions:

- Adding support for stereo sound. Currently everything is mono. One way we can implement stereo sound is to have nodes that can output stereo waves (2D numpy arrays) and nodes that can take stereo waves as input. We can also have nodes that can convert mono waves to stereo and vice versa. We can also have nodes that take in two mono waves and mix them into a stereo signal. We can also have nodes that can pan mono waves to stereo.
- More reallistic feedback loops with slight configurable delays to avoid infinite stacking of waves.
- Support for MIDI clock input and output, to be able to sync the playback with other devices.
- Support for MIDI note and cc output, to be able to control other devices with the midi node.
- Add a live audio input node, to be able to process live audio input from a microphone or line input.
- Being able to create new nodes or "macros" or "sub-patches" with parameters that can be reused in the yaml file.
- ✅ **IMPLEMENTED**: Expression system for dynamic evaluation of Python expressions in YAML (can set parameters as functions of time, variables, or other parameters)
- Musical note name constants (A4, C#5, etc.) and interval helpers (semitone, fifth, octave as frequency multipliers) for the expression system
- Beat/time division constants (beat, bar, eighth, sixteenth) based on BPM for the expression system

### Directives for AI

- Avoid saying things like "this likely happens because ...", do further checks to confirm your hypothesis.
- When possible we should avoid code repetition, for example if we are adding a new functionality or condition that is similar to an existing one, we should try to generalise the code to avoid repetition by for example creating a new function or class that encapsulates the common behaviour, and parameterising the differences.
- We don't need to be backwards compatible when making changes to nodes, I'm the only user for now, we can just update the yaml file as needed.
- We don't need to write detailed instructions of usage or changes in .md files, if there is anything non-trivial we can just add a comment on the node file, just above the node model class.
- For neatness, we try to keep node parameters as one word, but if we can't find a good name, we can use underscores to separate words. Feel free to suggest better names for parameters if you think of any.
- When passing positional arguments to functions or methods and the arguments have the same name as the variables, we should just pass them without specifying the name to avoid "name=name" argument passing.
- Very important:Use vectorised numpy operations when possible, avoid "for" loops over numpy arrays because this is a realtime application dealing with very large arrays.
- The timeout and gtimeout commands are not available on my system, avoid using them.
