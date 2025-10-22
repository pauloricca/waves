# Hot Reload & Auto-ID System Documentation Index

This document provides a quick reference to the hot reload and automatic ID generation system.

## Quick Links

### For Users
- **HIERARCHICAL_AUTO_IDS.md** - How automatic IDs work, when to use explicit IDs, and examples
- **waves.yaml** - Your sound definitions file that supports live editing

### For Node Developers
- **.github/copilot-instructions.md** - Instructions for creating new node types with hot reload support
- **nodes/sample.py** - Reference implementation of a stateful node

### For System Architects
- **HOT_RELOAD_IMPLEMENTATION_SUMMARY.md** - Complete technical overview
- **docs/HOT_RELOAD_PLAN.md** - Detailed architecture and design decisions
- **nodes/node_utils/** - Core implementation files

## Quick Facts

### Auto-ID System
- **Automatic**: All nodes get stable IDs without manual assignment
- **Hierarchical**: IDs reflect position in the tree (`"root.0.1"`, `"my_lfo.0"`)
- **Backwards compatible**: Explicit `id` fields still work and take priority
- **Integrated**: Works seamlessly with hot reload for state preservation

### Hot Reload
- **Live editing**: Edit waves.yaml while sounds are playing
- **State preservation**: Playback position, sample state, and parameters maintained
- **Thread-safe**: Safe to reload while rendering audio
- **Smart cleanup**: Removes state for deleted nodes to prevent memory leaks

## File Structure

```
waves/                                          # Project root
├── HIERARCHICAL_AUTO_IDS.md                   # Auto-ID guide (READ THIS FIRST)
├── HOT_RELOAD_IMPLEMENTATION_SUMMARY.md       # Technical overview
├── docs/
│   └── HOT_RELOAD_PLAN.md                     # Detailed architecture
├── .github/
│   └── copilot-instructions.md                # For node developers
├── nodes/
│   ├── sample.py                              # Reference stateful node
│   └── node_utils/
│       ├── auto_id_generator.py               # Auto-ID implementation
│       ├── hot_reload_manager.py              # State capture/restore
│       ├── instantiate_node.py                # Node creation with hot reload
│       └── base_node.py                       # Base class with ID support
└── waves.yaml                                 # Your sound definitions
```

## Reading Order

### For Sound Designers (Users)
1. **HIERARCHICAL_AUTO_IDS.md** - Learn how auto-IDs work (5 min)
2. **waves.yaml** - See examples in practice
3. Start editing sounds and watching hot reload work!

### For Node Developers
1. **HOT_RELOAD_IMPLEMENTATION_SUMMARY.md** - Auto-ID section (10 min)
2. **.github/copilot-instructions.md** - State preservation pattern (10 min)
3. **nodes/sample.py** - Study the reference implementation (10 min)
4. Create your stateful node following the pattern

### For Contributors
1. **docs/HOT_RELOAD_PLAN.md** - Architecture and design decisions (20 min)
2. **HOT_RELOAD_IMPLEMENTATION_SUMMARY.md** - Full implementation overview (20 min)
3. **nodes/node_utils/** - Study core components
4. **nodes/sample.py** - Reference implementation

## Key Concepts

### Automatic ID Generation
```yaml
my_sound:
  osc:                    # Auto ID: "root"
    id: my_lfo           # Override with explicit ID: "my_lfo"
    type: sin
    freq: 2
  delay:                  # Auto ID: "root.1"
    time: 0.3
    signal:
      reference:
        ref: my_lfo      # Can reference explicit ID
```

### State Preservation
```python
class MyNode(BaseNode):
    def __init__(self, model, state, hot_reload=False):
        super().__init__(model)
        self.state = state
        
        # Initialize on first load
        if not hot_reload:
            self.state.position = 0
        
        # Child nodes recreated each time
        self.child = create_child_node(model.param)
```

### Hot Reload Flow
```
1. Edit waves.yaml and save
2. File watcher detects change
3. On next audio chunk:
   - Capture state from current node tree
   - Reload and parse YAML
   - Instantiate new node tree
   - Restore state to matching nodes by ID
   - Continue playback from preserved position
4. Audio plays seamlessly with new definition
```

## Common Tasks

### I want to add a new parameter that survives reloads
→ Use explicit `id` on the node, or rely on auto-ID with no explicit ID

### I want to reference a node from another node
→ Give it an explicit `id`, then use `reference: {ref: my_id}`

### I want state preserved even without an ID
→ It's automatic now! All nodes get auto-generated IDs

### I want stable IDs when I reorder nodes
→ Use explicit IDs for important nodes, or accept auto-ID changes for reordering

### I need to debug why state isn't preserved
→ Check `DISPLAY_HOT_RELOAD_CLEANUP` in config.py to see state cleanup logs
→ Verify the node has either an explicit `id` or relies on auto-ID

## Troubleshooting

### State not preserved after editing
**Possible causes:**
- Node doesn't have an ID (explicit or auto-generated) → Now shouldn't happen, check logs
- Node type doesn't support `state` and `hot_reload` parameters → Update node to follow pattern
- Complex node tree restructuring changed auto-IDs → Use explicit IDs for stability

### Audio clicks/artifacts during hot reload
- Reload only happens at chunk boundaries
- RenderContext is cleared to prevent stale caches
- If issue persists, check node._do_render() for state discontinuities

### Missing state after changes
- **Check**: Does your node have a node_id? (should have auto-ID)
- **Check**: Is state being initialized in __init__? (check hot_reload=False path)
- **Check**: Are you accessing state correctly? (use self.state.attr_name)

## Configuration

In **config.py**:
```python
WAIT_FOR_CHANGES_IN_WAVES_YAML = True      # Enable hot reload
DISPLAY_HOT_RELOAD_CLEANUP = False         # Log cleanup of removed nodes' state
```

## Performance Notes

- Auto-ID generation happens at parse time (sound_library.py)
- Hierarchical ID building is O(n) where n = number of nodes
- State capture/restore is O(m) where m = number of nodes with IDs
- Hot reload doesn't happen during audio rendering (thread-safe)

## References

- Python SimpleNamespace documentation: https://docs.python.org/3/library/types.html#types.SimpleNamespace
- Pydantic models: https://docs.pydantic.dev/
- NumPy arrays: https://numpy.org/doc/stable/

## Summary

The waves project now has a complete hot reload system with automatic hierarchical ID generation:

✅ **Automatic state preservation** - No need to manually assign IDs to every node
✅ **Live YAML editing** - Edit sounds while they play without interruption  
✅ **Thread-safe** - Hot reload is safe during audio rendering
✅ **Backwards compatible** - Explicit IDs still work exactly as before
✅ **Memory efficient** - Orphaned state is automatically cleaned up

Start with HIERARCHICAL_AUTO_IDS.md to learn how to use it!
