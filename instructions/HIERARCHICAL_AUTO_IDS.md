# Parameter-Path Auto ID Generation

## Overview

The waves project automatically generates IDs for all nodes based on the **parameter path** where they're used in the node tree, enabling state preservation across hot reloads even for nodes without explicit IDs.

IDs are generated at **node instantiation time** based on:
- Parent node's ID
- Parameter name where the child is used
- Index if the parameter is a list

This means:

- **Explicit IDs**: `"my_oscillator"` (manually provided in YAML)
- **Auto-generated IDs**: `"root.freq"`, `"root.signals.0.WavableValueModel"`, `"root.signals.1.OscillatorModel"`
- **Benefits**: State preserved across hot reloads; IDs are stable and predictable

## ID Priority

IDs are determined in this priority order:

1. **Explicit ID** - If you provide `id: "my_node"` in YAML
2. **Auto-generated ID** - Hierarchical ID based on position in tree

Both types get state preservation during hot reloads!

## ID Format

### Based on Parameter Path

IDs reflect the **parameter name and list indices** where a node is connected:

```
Root node: "root"
Nested in parameter: "root.parameter_name"
In a list: "root.list_name.0", "root.list_name.1"
Multiple levels: "root.mix.signals.0", "root.delay.signal"
```

### Examples from waves.yaml:

```yaml
test-ids:
  osc:                          # Root: "root"
    freq:                       # Parameter path: root.freq
      osc:                      # Auto ID: "root.freq" (node at freq parameter)
        freq: 1
        range: [10, 50]

test-ids-2:
  mix:                          # Root: "root"
    signals:                    # Parameter path: root.mix.signals
      - osc:                    # Auto ID: "root.mix.signals.0"
          freq: 200
      - osc:                    # Auto ID: "root.mix.signals.1"
          freq: 400

my_sound:
  delay:                        # Root: "root"
    id: my_delay               # Explicit ID: "my_delay" (overrides auto ID)
    time: 0.3
    signal:                     # Parameter path within delay context: my_delay.signal
      osc:                      # Auto ID: "my_delay.signal"
        freq: 440
```

## Benefits

### 1. **State Moves With the Parameter**
State is tied to WHERE a node is used (the parameter), not its position in a list:

```yaml
# BEFORE - state at root.mix.signals.0
mix:
  signals:
    - osc:              # State: root.mix.signals.0
        freq: 200

# AFTER - add a new osc at the beginning
mix:
  signals:
    - osc:              # State: root.mix.signals.0 (NEW osc, fresh state)
        freq: 100
    - osc:              # State: root.mix.signals.1 (OLD osc state moved here!)
        freq: 200
```

**Result**: Old osc keeps its state (playback position, internal counters) even though it moved from index 0 to index 1.

### 2. **Adding/Removing Siblings Doesn't Break State**
Other nodes' IDs are unaffected when you add or remove siblings:

```yaml
SCENARIO: sequencer with 4 items, osc at index 3

Before:
  sequencer:
    sequence:
      - osc:          # ID: root.sequencer.sequence.0
      -
      -
      - osc:          # ID: root.sequencer.sequence.3 (state preserved here)

After: Add new item at index 1
  sequencer:
    sequence:
      - osc:          # ID: root.sequencer.sequence.0 (unchanged)
      - kick:         # ID: root.sequencer.sequence.1 (NEW)
      -
      -
      - osc:          # ID: root.sequencer.sequence.3 (STILL HERE! State preserved!)

The osc at index 3 keeps the same ID and its state is preserved!
```

### 3. **No State Confusion When Swapping Node Types**
Different node types have different internal state structures:

```yaml
# BEFORE - sample player has playhead_position
- sample:
    id: drum1          # Explicit ID: drum1
    file: kick.wav

# AFTER - swapped to osc (mistake!)
- osc:
    id: drum1          # Same ID, different node type!
    freq: 440

# What happens:
# Sample's state (playhead_position, total_samples_rendered)
# gets applied to Osc, which expects different state (phase, etc.)
# Result: potential errors or unexpected behavior

# SOLUTION - don't reuse explicit IDs across types!
# Or use auto-IDs which encode the parameter context
```

### 4. **Parameter Context Prevents Mistakes**
Auto-IDs include the parameter name, making them semantically clear:

```yaml
my_delay:
  delay:
    signal:           # Parameter: signal
      osc:           # Auto ID: my_delay.signal (clear it's the delay's signal!)
        freq: 440
```

This makes it obvious what each node is connected to, reducing confusion.

## Benefits Summary

| Scenario | Position-Based (Old) | Parameter-Path (New) |
|----------|---------------------|----------------------|
| Add sibling | All indices shift, state lost | Only new item affected, others preserved |
| Remove sibling | All indices shift, state lost | Other items unaffected |
| Reorder siblings | IDs change, state lost | IDs change but follow parameter path |
| Swap node types | State confusion possible | Auto-IDs unique per type, safer |

## Implementation Details

### instantiate_node() (`nodes/node_utils/instantiate_node.py`)

IDs are now generated **at instantiation time** using a simple hierarchical pattern:

```python
def instantiate_node(node_model_or_value, parent_id, attribute_name, attribute_index=None):
    # Generate ID based on parameter path
    if node_model_or_value.id is not None:
        node_id = node_model_or_value.id  # Explicit ID takes priority
    else:
        node_id = f"{parent_id}.{attribute_name}"
        if attribute_index is not None:
            node_id += f".{attribute_index}"
        node_id += f".{node_model_or_value.__class__.__name__}"
```

**How It Works:**
1. Every node receives its parent's ID, the attribute name, and optionally an index
2. IDs are built hierarchically: `parent_id.attribute_name[.index].ClassName`
3. Explicit IDs in YAML always take priority
4. For WavableValues (scalars, expressions), a random ID is generated

### BaseNode.instantiate_child_node()

Helper method that simplifies child node creation with proper ID generation:

```python
def instantiate_child_node(self, child, attribute_name, attribute_index=None):
    """Uses this node's ID as parent_id for stable runtime ID generation."""
    return instantiate_node(child, self.node_id, attribute_name, attribute_index)
```

## Usage Examples

### Example 1: State Follows Parameter Path
```yaml
my_composition:
  mix:
    signals:
      - sample:          # Auto ID: root.mix.signals.0
          file: kick.wav
      - sample:          # Auto ID: root.mix.signals.1
          file: snare.wav
```

When you edit and save:
- Both samples preserve playback state
- State is tied to the parameter path, not position
- If you reorder them, state moves with the parameter path

### Example 2: Nested Parameters
```yaml
my_sound:
  delay:
    id: echo            # Explicit ID: echo
    time: 0.5
    signal:             # Parameter path: echo.signal
      osc:              # Auto ID: echo.signal
        type: sin
        freq: 440
        phase:          # Parameter path: echo.signal.phase
          osc:          # Auto ID: echo.signal.phase
            type: sin
            freq: 2
```

State preserved at each level:
- `echo` - explicit ID, state preserved
- `echo.signal` - auto ID from parameter path
- `echo.signal.phase` - nested auto ID

### Example 3: Lists with Different Nodes
```yaml
test-ids-3:
  sequencer:
    interval: 5
    sequence:
      - osc:            # Auto ID: root.sequencer.sequence.0
          freq: 200
      -                 # Empty (index 1)
      -                 # Empty (index 2)
      - osc:            # Auto ID: root.sequencer.sequence.3
          freq: 400
```

Each osc keeps its state by parameter path:
- First osc: `root.sequencer.sequence.0`
- Second osc: `root.sequencer.sequence.3` (index preserved!)

### Example 4: Mixed Explicit and Auto IDs
```yaml
my_song:
  mix:
    signals:
      - sample:
          id: drums     # Explicit: drums
          file: loop.wav
      - osc:            # Auto: root.mix.signals.1 (no id provided)
          type: sin
          freq: 440
      - delay:
          id: echo      # Explicit: echo
          time: 0.5
          signal:
            reference:
              ref: drums  # Can reference by explicit ID
```

State preservation:
- `drums` - explicit ID, can be referenced
- `root.mix.signals.1` - auto ID for the osc
- `echo` - explicit ID, can be referenced

## Configuration

Optional: Enable debug logging for ID generation:
```python
# In config.py (future enhancement)
DEBUG_AUTO_ID_GENERATION = False  # Set to True to see generated IDs
```

## Gotchas & Notes

1. **Auto-Generated IDs Include Class Names**
   - Auto IDs include the model class name: `root.freq.OscillatorModel`
   - This makes IDs unique even when swapping node types at the same parameter
   - Different node types have different internal state, so this prevents state confusion

2. **List Index Changes When Items Are Added/Removed**
   - When you add an item to a list, items after it shift indices
   - Their auto-IDs change (e.g., `root.signals.0` → `root.signals.1`)
   - State is NOT automatically migrated - new indices get fresh state
   - Use explicit IDs if you need state to follow a specific node through restructuring

3. **Explicit IDs Must Be Unique**
   - Each explicit ID should be used only once in the sound tree
   - Using the same ID twice will cause the second node to overwrite the state of the first
   - Auto-IDs are always unique (based on full parameter path + class name)

4. **Explicit IDs Create New ID Hierarchy**
   - When you use explicit ID, children's IDs start from that ID
   - Example: `my_delay` → child at signal becomes `my_delay.signal.OscillatorModel`
   - This is intentional - explicit IDs are "local roots" for their subtrees

## References

- `nodes/node_utils/instantiate_node.py` - ID generation and node instantiation logic
- `nodes/node_utils/base_node.py` - BaseNode with instantiate_child_node() helper
- `nodes/node_utils/node_state_registry.py` - Global state management indexed by node ID
- `sound_library.py` - YAML parsing that creates the initial model tree
