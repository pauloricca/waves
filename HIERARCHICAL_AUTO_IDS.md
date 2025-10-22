# Parameter-Path Auto ID Generation

## Overview

The waves project now automatically generates IDs for all nodes based on the **parameter path** where they're connected, enabling state preservation across hot reloads even for nodes without explicit IDs.

IDs are generated based on WHERE a node is used (the parameter name), not WHERE it sits in the tree. This means:

- **Explicit IDs**: `"my_oscillator"` (manually provided in YAML)
- **Auto-generated IDs**: `"root.freq"`, `"root.mix.signals.0"`, `"root.mix.signals.1"`
- **Benefits**: State preserved when adding/removing/reordering siblings; no state confusion when swapping node types

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

### AutoIDGenerator (`nodes/node_utils/auto_id_generator.py`)

**Key Methods:**
- `generate_ids(obj, param_path="root")` - Recursively generate IDs based on parameter paths
- `get_effective_id(model)` - Get ID (explicit or auto-generated)
- `get_auto_id(model)` - Get only auto-generated ID

**How It Works:**
1. Traverses the entire YAML/model tree recursively
2. For each parameter, builds a path string (e.g., "root.mix.signals")
3. For list items, appends the index (e.g., "root.mix.signals.0")
4. If a node has no explicit `id`, stores the parameter path in `__auto_id__`
5. If a node has explicit `id`, uses that instead (takes priority)

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

1. **Swapping Node Types with Same Explicit ID**
   - ❌ Don't do this: Replace sample with osc but keep `id: drum1`
   - Different node types have different internal state
   - State from one type can't be used by another type
   - ✅ Solution: Use type-specific IDs or rely on auto-IDs

2. **List Index Changes When Items Are Added/Removed**
   - When you add an item to a list, items after it shift indices
   - Their auto-IDs change (e.g., `root.signals.0` → `root.signals.1`)
   - But their state still moves with them (tied to the parameter path)
   - ✅ This is actually good! State follows the node naturally

3. **Explicit IDs Must Be Unique**
   - Each explicit ID should be used only once in the sound tree
   - Using the same ID twice causes undefined behavior
   - ✅ Auto-IDs are always unique (tied to parameter path)

4. **Explicit IDs Break Parameter-Path Hierarchy**
   - When you use explicit ID, children's parameter paths start from that ID
   - Not from the original parameter path
   - This is intentional - explicit IDs are "local roots"
   - ✅ Good for organizing state around important nodes

## Future Enhancements

1. **Stable list indices**: Could use node types or other identifiers instead of numeric indices
2. **ID migration**: Support migration when tree structure changes
3. **ID validation**: Warn about ID collisions or instability
4. **Debug visualization**: Show generated IDs in CLI/UI

## References

- `nodes/node_utils/auto_id_generator.py` - ID generation implementation
- `sound_library.py` - Integration point for YAML loading
- `nodes/node_utils/instantiate_node.py` - Node instantiation with ID support
- `nodes/node_utils/hot_reload_manager.py` - State capture/restore with ID support
