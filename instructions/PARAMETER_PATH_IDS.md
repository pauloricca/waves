# Parameter-Path Auto ID System - Implementation Summary

## Overview

The waves project uses **parameter-path-based auto IDs** generated at node instantiation time. This means:

- **IDs are based on WHERE a node is used** (parent_id + parameter name + index)
- **IDs include class names** to avoid state confusion when swapping node types
- **Simple hierarchical pattern**: `parent_id.attribute_name[.index].ClassName`
- **Backwards compatible** with explicit IDs (they take priority)

## How It Works

### ID Generation

IDs are generated in `instantiate_node()` based on three parameters:
1. **parent_id**: The ID of the parent node
2. **attribute_name**: The parameter name where this node is used
3. **attribute_index**: The index if the parameter is a list (optional)

### ID Format

```python
# For nodes with explicit IDs:
node_id = model.id  # e.g., "my_oscillator"

# For nodes without explicit IDs:
node_id = f"{parent_id}.{attribute_name}"
if attribute_index is not None:
    node_id += f".{attribute_index}"
node_id += f".{model.__class__.__name__}"
# e.g., "root.freq.OscillatorModel" or "root.signals.0.SampleModel"
```

## Examples

```yaml
test-sound:
  osc:                          # Root: "root"
    freq:                       # Parameter: freq
      osc:                      # ID: "root.freq.OscillatorModel"
        type: sin
        freq: 1
        range: [10, 50]
```

```yaml
test-mix:
  mix:                          # Root: "root"
    signals:
      - osc:                    # ID: "root.signals.0.OscillatorModel"
          freq: 200
      - sample:                 # ID: "root.signals.1.SampleModel"
          file: kick.wav
      - osc:                    # ID: "root.signals.2.OscillatorModel"
          id: my_lfo            # Explicit ID takes priority: "my_lfo"
          freq: 2
## Implementation

IDs are generated in `nodes/node_utils/instantiate_node.py`:

```python
def instantiate_node(node_model_or_value, parent_id, attribute_name, attribute_index=None):
    # Generate ID
    if isinstance(node_model_or_value, (int, float, str, list)):
        node_id = random().hex()  # Random ID for primitives
    elif node_model_or_value.id is not None:
        node_id = node_model_or_value.id  # Explicit ID
    else:
        # Auto-generate based on parameter path
        node_id = f"{parent_id}.{attribute_name}"
        if attribute_index is not None:
            node_id += f".{attribute_index}"
        node_id += f".{node_model_or_value.__class__.__name__}"
    
    # Get or create state from global registry
    state = get_state_registry().get_state(node_id)
    if state is None:
        state = get_state_registry().create_state(node_id)
        old_state_existed = False
    else:
        old_state_existed = True
    
    # Instantiate with hot_reload flag
    return node_class(node_model, node_id, state, do_initialise_state=old_state_existed)
```

### Helper Method

`BaseNode.instantiate_child_node()` simplifies child creation:

```python
def instantiate_child_node(self, child, attribute_name, attribute_index=None):
    """Uses this node's ID as parent_id for proper hierarchical IDs."""
    return instantiate_node(child, self.node_id, attribute_name, attribute_index)
```

## State Behavior with List Changes

### Adding Items
When you add an item to a list, **indices shift** and IDs change:
```yaml
BEFORE:
  - osc:    # ID: root.signals.0.OscillatorModel
  - sample: # ID: root.signals.1.SampleModel

AFTER: Add new item at start
  - delay:  # ID: root.signals.0.DelayModel (NEW, fresh state)
  - osc:    # ID: root.signals.1.OscillatorModel (state LOST, new ID!)
  - sample: # ID: root.signals.2.SampleModel (state LOST, new ID!)
```

**Important**: State is NOT migrated automatically. If you need to preserve state when reordering, use explicit IDs.

### Removing Items
Similarly, removing items causes indices to shift:
```yaml
BEFORE:
  - osc:    # ID: root.signals.0.OscillatorModel
  - sample: # ID: root.signals.1.SampleModel
  - delay:  # ID: root.signals.2.DelayModel

AFTER: Remove middle item
  - osc:    # ID: root.signals.0.OscillatorModel (preserved)
  - delay:  # ID: root.signals.1.DelayModel (state LOST, was .2 before!)
```
  signals:
    - osc: freq=200   # ID: root.mix.signals.0 (state: playhead=1000)
    - osc: freq=400   # ID: root.mix.signals.1 (state: playhead=2000)

AFTER: Reorder
  signals:
    - osc: freq=400   # ID: root.mix.signals.0 (now has playhead=1000 state)
    - osc: freq=200   # ID: root.mix.signals.1 (now has playhead=2000 state)
```

Wait, that looks weird. Let me clarify...

Actually, when you reorder in YAML:
```yaml
BEFORE:
  - osc: freq=200   # ID and state: root.mix.signals.0 → playhead=1000
  - osc: freq=400   # ID and state: root.mix.signals.1 → playhead=2000

AFTER (reordered):
  - osc: freq=400   # ID and state: root.mix.signals.0 → (fresh state, this is different!)
  - osc: freq=200   # ID and state: root.mix.signals.1 → (fresh state, this is different!)
```

### Using Explicit IDs for Stable State

When you need state to survive list reordering, use explicit IDs:

```yaml
my_mix:
  mix:
    signals:
      - osc:
          id: bass_osc      # Explicit ID - state survives reordering
          freq: 100
      - osc:
          id: lead_osc      # Explicit ID - state survives reordering  
          freq: 440
      - sample:             # Auto ID - state tied to index
          file: kick.wav
```

Now you can reorder `bass_osc` and `lead_osc` and their state follows them!

## Key Characteristics

### Class Names in IDs
Auto-generated IDs include the model class name:
- `root.freq.OscillatorModel`
- `root.signals.0.SampleModel`
- `root.delay.signal.WavableValueModel`

This prevents state confusion when swapping different node types at the same parameter path.

### State and List Changes
**Important**: Auto-generated IDs change when list indices change. State does NOT migrate automatically.

```yaml
BEFORE:
  - osc:    # ID: root.signals.0.OscillatorModel (has state)
  - sample: # ID: root.signals.1.SampleModel (has state)

AFTER: Add item at start
  - delay:  # ID: root.signals.0.DelayModel (NEW, fresh state)
  - osc:    # ID: root.signals.1.OscillatorModel (NEW ID, loses old state!)
  - sample: # ID: root.signals.2.SampleModel (NEW ID, loses old state!)
```

Use explicit IDs if you need state to survive structural changes.

## When to Use What

| Scenario | Use |
|----------|-----|
| Simple, no references | Auto IDs (parameter-path based) |
| Need to reference from another node | Explicit ID |
| Want state to survive reordering | Explicit ID |
| Part of a sequence/list, don't care about state migration | Auto ID |
| Testing different node types at same parameter | Auto ID (class name prevents confusion) |

## Summary

The current parameter-path ID system:
1. **Generates IDs at instantiation time** based on parent_id + attribute_name + index + class_name
2. **Explicit IDs always take priority** over auto-generated IDs
3. **Class names prevent state confusion** when swapping node types
4. **Simple and predictable** - easy to understand the ID pattern
5. **Backwards compatible** - explicit IDs work exactly as before

This provides a good balance between automatic state preservation and user control.
