# Node String Notation

This feature allows you to instantiate nodes using a compact string notation with parameters embedded in the string.

## Syntax

The syntax is: `node_name param1VALUE param2VALUE ...`

For expression or variable values, use assignment syntax:
`node_name param=EXPRESSION`

Where:
- `node_name` is the name of the sound/node defined in waves.yaml
- Parameters are written as `paramNAMEVALUE` with no space between name and value
- Parameters can also be written as `param=EXPRESSION` to reference render/context variables
- In sequencer and automation steps, `prob` is reserved as a probability condition
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
play p0.5 s=note        # Play with p=0.5 and speed driven by the note variable
lead f=note*2           # Use an expression from the current render context
kick prob=0.5           # 50% chance to play this sequencer step
kick prob=cycle%4==0    # Play only on cycles 0, 4, 8...
```

## Usage

### 1. Command Line

You can pass parameters when running sounds from the command line:

```bash
./waves.py kick f440 a0.5
./waves.py --save kick f440 a0.5
./waves.py my_sound t2
./waves.py lead f880 a0.2
./waves.py hihat v2
```

Use `--save` to save realtime playback, equivalent to setting
`DO_RECORD_REAL_TIME = True` for that run.

### 2. In Sequencer

The sequencer supports this notation in its sequence/chain:

```yaml
my_sequence:
  sequencer:
    interval: 0.5
    sequence:
      - kick f440 a0.5
      - lead f880
      - play p0.5 s=note
      - hihat prob=cycle%2==0
      - [kick f200, lead f400]  # Multiple sounds in one step
      - hihat v2
```

### 3. In Automation

Automation string steps can also use the reserved `prob` parameter. A skipped
automation step behaves like an empty step, so the previous active value holds.
The `cycle` variable starts at `0` and increments each time the automation loops.

```yaml
freq:
  automation:
    interval: 0.25
    repeat: 9999
    steps:
      - C3
      - E3 prob=cycle%2==0
      - G3 prob=0.5
      - B3 prob=false
```

### 4. Sub-Patching (YAML Node References)

You can reference other sounds defined in the YAML file as if they were node types (as long as they have is_reusable: true, so we don't polute the schema with all the sounds), and apply parameters to them:

```yaml
# Define sounds
hihat:
  context:
    is_reusable: true
    v: 1
    signal:
      # ... hihat definition

kick:
  mix:
    is_reusable: true
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
