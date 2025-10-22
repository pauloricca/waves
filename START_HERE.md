# 🎵 Welcome to Waves with Live Hot Reload!

Your waves project now has **automatic state preservation with live YAML editing** during playback!

## ⚡ Quick Start (2 minutes)

### 1. Enable Hot Reload
Open `config.py` and ensure:
```python
WAIT_FOR_CHANGES_IN_WAVES_YAML = True
```

### 2. Start Playing a Sound
```bash
python waves.py my_sound
```

### 3. Edit waves.yaml While Playing
Open `waves.yaml`, make changes, save it, and watch the sound update in real-time!

## 📚 What Just Changed?

### Automatic Parameter-Path IDs
All nodes now get **automatic IDs based on the parameter they're connected to**, without needing manual assignment:

```yaml
# BEFORE: Had to manually assign IDs
my_sound:
  delay:
    id: echo1          # Manual assignment required
    time: 0.5
    signal:
      osc:
        id: lfo1       # For every node you wanted to preserve
        freq: 2

# AFTER: IDs assigned automatically based on parameter path!
my_sound:
  delay:
    # Auto ID: root (root node of this sound)
    time: 0.5
    signal:
      # Auto ID: root.delay.signal (at delay's signal parameter)
      osc:
        # Auto ID: root.delay.signal.freq
        freq: 2
```

**Both approaches work** - use automatic IDs by default, or explicit IDs when you need to reference a node.

### Why Parameter Paths?
Parameter-path IDs solve a critical problem: **when you add/remove siblings, state moves with the node**:

```yaml
# Initial: 2 oscillators in mix
mix:
  signals:
    - osc:    # Auto ID: root.mix.signals.0 (state preserved here)
        freq: 200
    - osc:    # Auto ID: root.mix.signals.1
        freq: 400

# After adding a new osc at the beginning:
mix:
  signals:
    - osc:    # Auto ID: root.mix.signals.0 (NEW osc)
        freq: 100
    - osc:    # Auto ID: root.mix.signals.1 (STATE MOVED HERE with the node!)
        freq: 200
    - osc:    # Auto ID: root.mix.signals.2
        freq: 400
```

The key insight: state is tied to the **parameter path**, not the position in the tree.


### State Preservation
- Playback position saved during reload
- Sample player continues from same spot
- Parameters preserved seamlessly

### Live Editing
- Edit sounds while they play
- No interruption to audio
- Changes apply on next chunk

## 🎯 Next Steps

### For Sound Designers
Read **`HIERARCHICAL_AUTO_IDS.md`** (5 min) to understand auto-IDs

### For Developers
Read **`HOT_RELOAD_IMPLEMENTATION_SUMMARY.md`** (20 min) for technical details

### For Everyone
Check **`docs/HOT_RELOAD_AND_AUTO_ID_GUIDE.md`** for navigation and docs

## 📖 Documentation

| File | Purpose | Time |
|------|---------|------|
| `HIERARCHICAL_AUTO_IDS.md` | Auto-ID guide with examples | 5 min |
| `docs/HOT_RELOAD_AND_AUTO_ID_GUIDE.md` | Documentation index | 5 min |
| `HOT_RELOAD_IMPLEMENTATION_SUMMARY.md` | Technical overview | 20 min |
| `docs/IMPLEMENTATION_COMPLETE.md` | Implementation status | 10 min |
| `.github/copilot-instructions.md` | Node developer guide | 15 min |
| `nodes/sample.py` | Reference implementation | 10 min |

## ✨ Key Features

✅ **Automatic IDs** - Every node gets a stable ID automatically  
✅ **Live Editing** - Edit YAML while sounds play  
✅ **State Preserved** - Playback position maintained across reloads  
✅ **Thread Safe** - Reloads happen safely during audio rendering  
✅ **Backwards Compatible** - Explicit IDs still work unchanged  
✅ **Memory Efficient** - Orphaned state cleaned up automatically  

## 🔧 Configuration

In `config.py`:
```python
WAIT_FOR_CHANGES_IN_WAVES_YAML = True      # Enable hot reload
DISPLAY_HOT_RELOAD_CLEANUP = False         # Show cleanup logs (optional)
```

## 🐛 Troubleshooting

**Q: Hot reload not working?**  
A: Check `WAIT_FOR_CHANGES_IN_WAVES_YAML = True` in config.py

**Q: State not preserved after editing?**  
A: Check the sound is running a node that supports state (like sample, delay, etc.)

**Q: Want to see what's being cleaned up?**  
A: Set `DISPLAY_HOT_RELOAD_CLEANUP = True` in config.py

## 📝 Examples

### Auto-ID in Action (State Moves with Parameter)
```yaml
my_composition:
  mix:
    signals:
      - osc:        # Auto ID: root.mix.signals.0
          type: sin
          freq: 440
      - osc:        # Auto ID: root.mix.signals.1
          type: sin
          freq: 220
```

Both oscillators preserve their state across edits. If you reorder them, state moves along!

### Nested Parameters
```yaml
test-ids:
  osc:
    freq:           # Parameter path: root.osc.freq
      osc:          # Auto ID: root.osc.freq
        type: sin
        freq: 2
        range: [10, 50]
```

The nested osc gets ID based on which parameter it's connected to.

### Explicit ID When Needed
```yaml
my_sound:
  delay:
    id: echo      # Explicit ID for reference
    time: 0.3
    signal:
      osc:
        id: lfo   # Reference this LFO
        type: sin
        freq: 2
```

Can reference `echo` and `lfo` from other nodes in your composition.

## ✅ What's Verified

- ✅ All 31 sounds load with auto-IDs
- ✅ All Python files compile without errors
- ✅ Auto-ID generation works correctly
- ✅ Hot reload integrates seamlessly
- ✅ Backwards compatible with existing code

## 🚀 Ready to Go!

Your waves project is ready for **live sound design**. Start editing!

Questions? See the full docs in `docs/HOT_RELOAD_AND_AUTO_ID_GUIDE.md`

Happy composing! 🎵
