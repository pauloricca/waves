# Hot Reload & Live State Preservation Plan

## Overview
This document outlines the plan for implementing live hot-reloading of the YAML sound definition file in the `waves` project, with seamless state preservation for nodes during playback. The goal is to allow editing and saving the YAML file while audio is playing, with changes reflected in real time, without interrupting playback or losing node state.

---

## Key Concepts

### 1. Node State as Simple Object
- Each node receives a `state` object in its `__init__` method and assigns it to `self.state`.
- The `state` object is a plain Python object (e.g., `types.SimpleNamespace`), not a dict or tuple.
- State variables are accessed as attributes: `self.state.last_playhead_position` (not as dict keys or tuple indices).
- The `state` object is only passed to `__init__`, not to `_do_render` or `render`—nodes always access state via `self.state`.
- There is no `node_state.py` file or helper class; state is just a simple object with attributes.

### 2. State Management Pattern
- On node creation, if not hot reloading, initialize all required state attributes on `self.state`.
- On hot reload, do not re-initialize state; the state object is already populated.
- Never use string keys (e.g., `self.state['foo']`) or tuple unpacking for state—always use attribute access.
- Never pass state to `_do_render` or `render`—always use `self.state`.

### 3. Example
```python
from types import SimpleNamespace

class SampleNode(BaseNode):
    def __init__(self, model, state, hot_reload=False):
        super().__init__(model)
        self.model = model
        self.state = state
        # Only persistent playback state in self.state
        if not hot_reload:
            self.state.last_playhead_position = 0
            self.state.total_samples_rendered = 0
        # All child nodes/helpers are regular attributes
        self.chop_node = wavable_value_node_factory(model.chop)
        self.speed_node = wavable_value_node_factory(model.speed)
        # ...
    def _do_render(self, ...):
        # Use persistent state
        self.state.last_playhead_position += 1
        # Use child node
        value = self.chop_node.render(...)
```

#### Only persistent playback state in self.state
- Do NOT put child nodes, wavable value nodes, or helpers in self.state.
- All usages in methods should reference child nodes as self.child_node, not self.state.child_node.

#### Example (incorrect):
```python
# Don't do this:
self.state.chop_node = wavable_value_node_factory(model.chop)
self.state.speed_node = wavable_value_node_factory(model.speed)
```

**This pattern must be followed for all nodes to ensure correct hot reload behavior and avoid subtle bugs.**

### 4. HotReloadManager
The `HotReloadManager` (nodes/node_utils/hot_reload_manager.py) enables state preservation across reloads:

- **capture_state(node)** - Recursively traverses the node tree and captures persistent state from all nodes with ids. Returns a dictionary mapping node IDs to their state objects.
- **restore_state(node, state_dict)** - Recursively traverses a newly instantiated node tree and restores captured state to matching nodes (matched by id).
- **get_all_node_ids(node)** - Returns a set of all node IDs in the tree (used to determine which nodes existed before reload).

Usage flow:
```python
manager = HotReloadManager()
# Before reload
old_state = manager.capture_state(old_node_tree)
old_ids = manager.get_all_node_ids(old_node_tree)

# Reload YAML and instantiate new tree with hot_reload flags
new_node_tree = instantiate_node_tree(new_model, hot_reload=True, previous_ids=old_ids)

# Restore state to new tree
manager.restore_state(new_node_tree, old_state)
```

### 5. Node Instantiation Pipeline
Updated `instantiate_node()` (nodes/node_utils/instantiate_node.py) now supports:
- `hot_reload` parameter: Indicates this is a hot reload scenario
- `previous_ids` parameter: Set of node IDs from the previous tree
- Automatic state object creation for nodes with ids
- Backward compatibility: Falls back to old constructor signature for nodes that don't support state

A new `instantiate_node_tree()` convenience wrapper is available for root-level node creation.

### 6. Threaded Architecture
- Playback and YAML file watching run in separate threads.
- A lock ensures thread-safe updates to the model tree and state during reload.
- When YAML changes: capture state, reload YAML, instantiate new tree, restore state, resume playback.

### 7. Controlled Initialization
- Nodes receive a `hot_reload` flag in their constructor.
- If `hot_reload=True`, nodes skip state initialization (state will be restored).
- If `hot_reload=False`, nodes initialize state as usual.
- New nodes (not present in the previous tree) always initialize state.

---

## Implementation Steps & Status

### ✅ Completed
1. **NodeState Class** - Deprecated and removed (SimpleNamespace pattern is cleaner)
2. **SampleNode Refactored** - Reference implementation with correct state pattern
3. **HotReloadManager** - Implemented with capture/restore/get_all_node_ids methods
4. **Node Instantiation Pipeline** - Updated with hot_reload and previous_ids support

### 🔄 In Progress
- (Current task: HotReloadManager completed)

### ⏳ Pending
1. **Threaded Playback & Watcher** - Refactor waves.py to run playback and YAML watcher in separate threads
2. **Documentation & Testing** - Document hot-reload API and test with various node types

---

## Usage Instructions

- Edit and save `waves.yaml` while playback is running.
- The system will detect changes, reload the YAML, and update the model tree in real time.
- Node state is preserved for matching nodes (by id); new nodes are initialized as usual.
- For best results, assign explicit `id` fields to nodes you want to preserve across edits.

---

## References
- This plan is referenced in code comments and PRs as `docs/HOT_RELOAD_PLAN.md`.
- SampleNode in `nodes/sample.py` is the reference implementation of the state pattern.
- HotReloadManager in `nodes/node_utils/hot_reload_manager.py` handles state capture/restore.

---

## Next Steps
- Implement threaded playback and file watching in `waves.py`.
- Add hot reload orchestration to detect YAML changes and coordinate reload.
- Test with various node types and YAML editing scenarios.
