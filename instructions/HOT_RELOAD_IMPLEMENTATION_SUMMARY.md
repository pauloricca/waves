# Hot Reload Implementation Summary

## Overview
Successfully implemented a complete hot-reload system for the waves project, enabling live YAML editing during playback without interrupting audio. The system preserves node state across reloads using a global state registry indexed by parameter-path based node IDs.

## Architecture Components

### 1. **Global State Registry** (`nodes/node_utils/node_state_registry.py`)
Centralized state management for all nodes.

**Key Methods:**
- `get_state(node_id)` - Returns existing state or None
- `create_state(node_id)` - Creates and stores new SimpleNamespace for a node
- `clear()` - Clears all states (for testing/cleanup)

**How it works:**
- Global singleton stores state objects indexed by node ID
- State persists across hot reloads automatically
- On reload, nodes with the same ID get their previous state back

### 2. **Enhanced instantiate_node()** (`nodes/node_utils/instantiate_node.py`)
Central function for node instantiation with automatic ID generation and state management.

**Signature:**
```python
def instantiate_node(node_model_or_value, parent_id, attribute_name, attribute_index=None)
```

**Key Features:**
- Generates hierarchical IDs based on parameter path: `parent_id.attribute_name[.index].ClassName`
- Explicit IDs in models take priority over auto-generated IDs
- Retrieves or creates state from global registry for each node
- Passes `hot_reload=True` to node constructor if state existed before

### 3. **State Management Pattern**
All stateful nodes follow a consistent pattern for hot reload support.

**Node Implementation Pattern:**
```python
class MyNode(BaseNode):
    def __init__(self, model, node_id: str, state, do_initialise_state=True):
        super().__init__(model, node_id, state, do_initialise_state)
        self.model = model
        # self.state is assigned in BaseNode.__init__
        
        # Initialize persistent state only if not hot reloading
        if do_initialise_state:
            self.state.playhead_position = 0
            self.state.samples_rendered = 0
        
        # Child nodes are regular attributes (recreated each time)
        self.child_node = self.instantiate_child_node(model.child, "child")
```

**Key Rules:**
1. Accept `node_id: str`, `state` and `hot_reload` parameters in `__init__`
2. Call `super().__init__(model, node_id, state, do_initialise_state)` first
3. Only store persistent state in `self.state` (playback position, counters)
4. Child nodes, helpers, and ephemeral data are regular attributes
5. Initialize state attributes only when `not hot_reload`
6. Use `self.instantiate_child_node()` to create child nodes with proper IDs
4. Use attribute access: `self.state.attr_name` (not dict keys)
5. Always give important nodes an `id` field for state preservation

### 4. **Enhanced waves.py** (`waves.py`)
Main playback controller with hot reload support.

**Key Components:**
- `hot_reload_lock` - Threading lock for thread-safe state updates
- `current_sound_node` - Tracks currently playing node for reloads
- `yaml_changed` - Flag signaling YAML file changes

**Hot Reload Flow:**
```
1. File watcher detects YAML change → sets yaml_changed = True
2. Audio callback checks yaml_changed flag
3. If true, re-instantiates the entire node tree
4. State is preserved automatically via global registry:
   - Nodes with same IDs get their previous state back
   - New nodes get fresh state
   - Removed nodes' state remains in registry (not cleaned up)
5. Playback continues with new tree and preserved state
```

**Modified play_in_real_time():**
- Tracks `active_sound_node` locally (can be updated during hot reload)
- Audio callback checks for YAML changes each chunk
- Thread-safe state updates via `hot_reload_lock`
- Resets RenderContext after hot reload to clear caches

### 5. **Reference Implementation: SampleNode** (`nodes/sample.py`)
Serves as the canonical example of hot reload support.

**State Management:**
- `state.last_playhead_position` - Current playback position
- `state.total_samples_rendered` - Total samples output
- `state.current_chop_index` - Active chop file index

**Ephemeral Attributes:**
- `speed_node, freq_node, start_node, end_node, offset_node, chop_node` - Child nodes
- `audio` - Currently loaded audio buffer (recreated on chop change)
- `_chop_files` - Cached directory listing

## Usage Example

### YAML Setup
```yaml
my_sound:
  sample:
    id: main_sample         # Important: give node an ID
    file: samples/kick.wav
    speed: 1.0
    duration: 2
```

### Editing Workflow
1. Start playback: `python waves.py my_sound`
2. Edit `waves.yaml` while playing (e.g., change `speed: 2.0`)
3. Save file → system detects change
4. On next audio chunk: state captured, YAML reloaded, new tree instantiated, state restored
5. Playback continues from preserved position with new parameters

## Technical Details

### State Preservation Mechanism
- All nodes get state objects from the global registry (indexed by node ID)
- Explicit IDs take priority; otherwise hierarchical IDs are auto-generated
- Each state object is a `SimpleNamespace` with arbitrary attributes
- File caching (`utils.py`) prevents unnecessary I/O when nodes recreate references
- RenderContext is reset after reload to clear audio cache
- State accumulates over time (not automatically cleaned up for now)

### Thread Safety
- `hot_reload_lock` protects access to `current_sound_node` during reload
- Audio callback holds lock while performing hot reload
- Lock acquisition is brief (just state operations, not rendering)

## Files Modified/Created

**Created:**
- `nodes/node_utils/hot_reload_manager.py` - State capture/restore manager
- `HOT_RELOAD_IMPLEMENTATION_SUMMARY.md` - This document

**Modified:**
- `nodes/node_utils/instantiate_node.py` - Hot reload support in node instantiation
- `waves.py` - Hot reload orchestration and state updates
- `docs/HOT_RELOAD_PLAN.md` - Updated documentation
- `.github/copilot-instructions.md` - Added hot reload section for node authors

**Reference Implementation:**
- `nodes/sample.py` - Already implemented with state pattern (reference)

## Configuration

Enable hot reload in `config.py`:
```python
WAIT_FOR_CHANGES_IN_WAVES_YAML = True  # Enable YAML file watching
```

## Testing Recommendations

1. **Basic Reload:**
   - Start playback with a sample node
   - Edit YAML to change a parameter
   - Verify smooth reload without audio interruption

2. **State Preservation:**
   - Use a sample player with looping or long audio
   - Edit YAML and verify playhead continues from same position
   - Verify audio buffer switches as expected

3. **Multiple Node IDs:**
   - Create composition with multiple nodes having explicit IDs
   - Edit different node parameters
   - Verify state preserved for all nodes

4. **Auto-Generated IDs:**
   - Create nodes without explicit IDs
   - Edit YAML and verify state is still preserved
   - Check that IDs are stable across minor edits

## References

- `nodes/sample.py` - Reference node implementation
- `nodes/node_utils/instantiate_node.py` - ID generation and instantiation
- `nodes/node_utils/node_state_registry.py` - Global state management
- `nodes/node_utils/base_node.py` - BaseNode with instantiate_child_node()
- `.github/copilot-instructions.md` - API documentation for node authors
- `HIERARCHICAL_AUTO_IDS.md` - Detailed auto-ID system documentation
