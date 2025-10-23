# Hot Reload Implementation Summary

## Overview
Successfully implemented a complete hot-reload system for the waves project, enabling live YAML editing during playback without interrupting audio. The system preserves node state while allowing real-time sound definition updates.

## Architecture Components

### 1. **HotReloadManager** (`nodes/node_utils/hot_reload_manager.py`)
Manages state capture and restoration across YAML reloads.

**Key Methods:**
- `capture_state(node)` - Recursively captures persistent state from all nodes with IDs
- `restore_state(node, state_dict)` - Recursively restores captured state to matching nodes
- `get_all_node_ids(node)` - Returns set of all node IDs in the tree

**How it works:**
```
1. Traverse entire node tree recursively
2. For each node with an `id` field, capture its `self.state` object
3. Store mapping of node_id → state object
4. During restore, match nodes by ID and restore their state
```

### 2. **Enhanced instantiate_node Pipeline** (`nodes/node_utils/instantiate_node.py`)
Updated to support hot reload context and state initialization.

**Key Changes:**
- Added `hot_reload: bool` parameter - indicates this is a hot reload scenario
- Added `previous_ids: Set[str]` parameter - IDs from the previous tree
- Automatic creation of `SimpleNamespace` state objects for nodes with IDs
- New `instantiate_node_tree()` convenience wrapper for root-level instantiation

**State Initialization Logic:**
```python
node_hot_reload = hot_reload and (obj.id in previous_ids if obj.id else False)
state = SimpleNamespace() if (obj.id) else None

# Try calling with state/hot_reload (new pattern)
node_definition.node(obj, state=state, hot_reload=node_hot_reload)
# Fall back to old pattern if not supported
```

### 3. **State Management Pattern**
All stateful nodes follow a consistent pattern for hot reload support.

**Node Implementation Pattern:**
```python
class MyNode(BaseNode):
    def __init__(self, model, state, hot_reload=False):
        super().__init__(model)
        self.model = model
        self.state = state
        
        # Initialize persistent state only if not hot reloading
        if not hot_reload:
            self.state.playhead_position = 0
            self.state.samples_rendered = 0
        
        # Child nodes are regular attributes (recreated each time)
        self.speed_node = wavable_value_node_factory(model.speed)
        self.freq_node = wavable_value_node_factory(model.freq)
```

**Key Rules:**
1. Accept `state` and `hot_reload` parameters in `__init__`
2. Only store persistent state in `self.state` (playback position, counters)
3. Child nodes, helpers, and ephemeral data are regular attributes
4. Use attribute access: `self.state.attr_name` (not dict keys)
5. Always give important nodes an `id` field for state preservation

### 4. **Enhanced waves.py** (`waves.py`)
Refactored for hot reload support during real-time playback.

**Key Additions:**
- `hot_reload_manager` - Global instance of HotReloadManager
- `hot_reload_lock` - Threading lock for thread-safe state updates
- `current_sound_node` - Tracks currently playing node for reloads
- `yaml_changed` - Flag signaling YAML file changes
- `perform_hot_reload()` - Orchestrates the full reload cycle with state cleanup

**Hot Reload Flow:**
```
1. File watcher detects YAML change → sets yaml_changed = True
2. Audio callback checks yaml_changed flag
3. If true, acquires lock and calls perform_hot_reload()
4. perform_hot_reload():
   - Captures state from current node tree
   - Reloads YAML and instantiates new tree
   - Restores captured state to matching nodes
   - Identifies orphaned nodes (present before, removed after)
   - Cleans up state entries for removed nodes (memory hygiene)
   - Updates current_sound_node reference
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
- Only nodes with explicit `id` fields have their state preserved
- Each state object is a `SimpleNamespace` with arbitrary attributes
- File caching (`utils.py`) prevents unnecessary I/O when nodes recreate references
- RenderContext is reset after reload to clear audio cache
- **Orphaned state cleanup**: State entries for removed nodes are automatically deleted to prevent memory leaks
  - Compares node IDs before reload (old_ids) with node IDs after instantiation (new_ids)
  - Removes state entries for nodes in old_ids - new_ids
  - Can optionally log cleanup via `DISPLAY_HOT_RELOAD_CLEANUP` config

### Thread Safety
- `hot_reload_lock` protects access to `current_sound_node` during reload
- Audio callback holds lock while performing hot reload
- Lock acquisition is brief (just state operations, not rendering)

### Backward Compatibility
- Old nodes without state/hot_reload support still work
- `instantiate_node()` tries new pattern, falls back to old signature
- Graceful degradation if hot reload fails (continues with old node)

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
   - Edit `waves.yaml` to change a parameter
   - Verify smooth reload without audio interruption

2. **State Preservation:**
   - Use a sample player with `chop: true`
   - Edit YAML and verify playhead continues from same position
   - Verify audio buffer switches as expected

3. **Multiple Node IDs:**
   - Create composition with multiple nodes having explicit IDs
   - Edit different node parameters
   - Verify state preserved for all nodes

4. **New vs Existing Nodes:**
   - Add a new node to existing tree
   - Verify new node initializes state (not hot reloaded)
   - Verify existing nodes preserve their state

## Auto ID Generation

### 6. **AutoIDGenerator** (`nodes/node_utils/auto_id_generator.py`)
Automatically generates hierarchical IDs for all nodes, eliminating the need for manual `id` assignments while maintaining state preservation.

**Key Features:**
- **Explicit IDs have priority**: If you provide `id: "my_node"` in YAML, that's used
- **Auto-generated IDs for others**: Nodes without explicit IDs get hierarchical IDs like "0", "0.1", "my_lfo.0"
- **Recursive tree traversal**: Processes entire node hierarchy during sound library loading
- **Integration at parse time**: IDs are generated immediately after YAML parsing

**ID Format:**
- Root node: `"root"` or `"0"` (depending on context)
- Child at index 0: `"0"` or `"parent_id.0"`
- Nested children: `"0.1.2"` or `"my_lfo.0.child_name"`

**Example:**
```yaml
my_sound:
  mix:
    signals:
      - osc:           # Auto ID: "0" → "root.0"
          type: sin
          freq: 440
      - osc:           # Auto ID: "1" → "root.1"
          id: my_lfo   # Explicit: "my_lfo" (takes priority)
          type: sin
          freq: 2
```

**Key Methods:**
- `generate_ids(obj, parent_id="", index_in_parent=None)` - Recursively assign auto IDs
- `get_effective_id(model)` - Returns explicit ID or auto-generated ID
- `get_auto_id(model)` - Returns only auto-generated ID
- `_build_id(parent_id, index, node_name)` - Constructs hierarchical ID string

**How it works:**
1. After YAML is parsed into models in `sound_library.py`
2. `AutoIDGenerator.generate_ids()` is called on each sound model
3. Recursively traverses the model tree (handles dicts, lists, Pydantic models)
4. For each node without explicit ID, generates and stores auto ID in `__auto_id__` attribute
5. During instantiation/hot reload, `get_effective_id()` returns the appropriate ID

**Benefits:**
- ✅ State preservation without manual IDs
- ✅ Automatic ID stability for added/removed nodes with explicit IDs
- ✅ Works seamlessly with explicit IDs (explicit takes priority)
- ✅ Supports deeply nested structures and lists
- ✅ No breaking changes (explicit IDs work exactly as before)

**See Also:** `HIERARCHICAL_AUTO_IDS.md` for detailed usage guide

## Future Enhancements

1. **Stable list indices:** Use node types or identifiers instead of numeric indices
2. **Selective Reload:** Option to reload only specific branches of the node tree
3. **State Versioning:** Handle state migrations when node model changes
4. **Reload History:** Keep track of reload history for debugging
5. **Performance Metrics:** Log hot reload timing and impact on audio
6. **ID visualization:** Show generated IDs in CLI/UI for debugging

## References

- `docs/HOT_RELOAD_PLAN.md` - Detailed architecture documentation
- `nodes/sample.py` - Reference node implementation
- `nodes/node_utils/hot_reload_manager.py` - State manager implementation
- `.github/copilot-instructions.md` - API documentation for node authors
