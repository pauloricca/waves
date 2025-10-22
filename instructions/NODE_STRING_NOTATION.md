# Node String Notation

This feature allows you to instantiate nodes using a compact string notation with parameters embedded in the string.

## Syntax

The syntax is: `node_name param1VALUE param2VALUE ...`

Where:
- `node_name` is the name of the sound/node defined in waves.yaml
- Parameters are written as `paramNAMEVALUE` with no space between name and value
- Common parameter shortcuts:
  - `f` = `freq` (frequency)
  - `a` = `amp` (amplitude)
  - `t` = time/duration (depending on node)
  - `v` = variable (context-specific)

## Examples

```
kick f440 a0.5          # Play kick with frequency=440, amplitude=0.5
my_sound t2 f880        # Play my_sound with t=2, freq=880
lead f220               # Play lead with frequency=220
hihat v2                # Play hihat with v=2
```

## Usage

### 1. Command Line

You can pass parameters when running sounds from the command line:

```bash
./waves.py kick f440 a0.5
./waves.py my_sound t2
./waves.py lead f880 a0.2
./waves.py hihat v2
```

### 2. In Sequencer

The sequencer supports this notation in its sequence/chain:

```yaml
my_sequence:
  sequencer:
    interval: 0.5
    sequence:
      - kick f440 a0.5
      - lead f880
      - [kick f200, lead f400]  # Multiple sounds in one step
      - hihat v2
```

### 3. Sub-Patching (YAML Node References)

You can reference other sounds defined in the YAML file as if they were node types, and apply parameters to them:

```yaml
# Define sounds
hihat:
  context:
    v: 1
    signal:
      # ... hihat definition

kick:
  mix:
    signals:
      # ... kick definition

# Use them as sub-patches with parameters
my_sound:
  mix:
    signals:
      - hihat:  # Reference to hihat sound
          v: 2  # Override the v parameter
      - kick:
          amp: 0.5  # Override amplitude
      - hihat:
          v: 0.3  # Another hihat instance with different params
```

Parameters passed to sub-patches are applied directly to the referenced sound's model. Only parameters that exist in the original sound definition will be applied.

## Implementation

The string parsing functionality is centralized in `nodes/node_utils/node_string_parser.py` with these main functions:

- `parse_params_from_string(param_string)` - Parse parameter key-value pairs
- `parse_node_string(node_string)` - Parse full node string into name and params
- `apply_params_to_model(model, params)` - Apply parameters to a node model
- `instantiate_node_from_string(node_string, model)` - Complete instantiation from string

This approach allows for:
- Consistent parameter parsing across the codebase
- Easy extension to new contexts (MIDI, OSC, etc.)
- Future sub-patching support with minimal changes
