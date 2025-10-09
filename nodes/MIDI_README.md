# MIDI Node Documentation

## Quick Start

```bash
python waves.py midi-test
```

Connect your MIDI controller and start playing! 🎹

## Overview

The MIDI node enables live MIDI input for real-time sound synthesis. It listens to MIDI note-on events on a specific channel and triggers sounds with automatic pitch and velocity mapping.

**Key Features:**
- ✅ Polyphonic - Play multiple notes simultaneously
- ✅ Velocity sensitive - MIDI velocity controls amplitude (0-127 → 0.0-1.0)
- ✅ Channel filtering - Listen to specific MIDI channels (0-15)
- ✅ Flexible signal routing - Use any node as the sound source
- ✅ Real-time ready - Designed for live performance
- ✅ Thread-safe - MIDI events processed safely in real-time

## Parameters

- **channel** (int): MIDI channel to listen to (0-15). Default: 0
- **signal** (BaseNodeModel): The sound/node to trigger. Can be any node (oscillator, filter chain, etc.)

## How It Works

1. The MIDI node opens the first available MIDI input port (or creates a virtual port if none are available)
2. When a MIDI note-on event is received on the specified channel:
   - The MIDI note number is converted to a frequency (using standard MIDI tuning: A4 = 440Hz)
   - A new instance of the signal node is created
   - The frequency is passed to the signal via the `frequency` render parameter
   - The MIDI velocity is converted to an amplitude multiplier (0-127 → 0.0-1.0)
3. Multiple notes can be played simultaneously (polyphonic)
4. Notes play for their full duration (based on the signal's duration parameter)
5. The MIDI node never "finishes" - it continuously runs and waits for MIDI events

## MIDI Port Setup

### Automatic Port Detection
On startup, the MIDI node:
1. Scans for available MIDI input ports
2. Opens the first available port automatically
3. Falls back to creating a virtual port "waves_virtual" if none found

### macOS Setup
To create virtual MIDI ports:
1. Open **Audio MIDI Setup** app
2. Window → Show MIDI Studio
3. Double-click "IAC Driver"
4. Check "Device is online"
5. Add buses as needed

### Connecting Hardware
Simply connect your MIDI controller via USB before running the program.

## Configuration Examples

### Basic Sine Synth
```yaml
midi-test:
  midi:
    channel: 0
    signal:
      osc:
        type: sine
        duration: 1.0
        amp: 0.5
        attack: 0.01
        release: 0.3
```

### Triangle Synth with Longer Envelope
```yaml
midi-synth:
  midi:
    channel: 0
    signal:
      osc:
        type: tri
        duration: 2.0
        amp: 0.8
        attack: 0.05
        release: 0.5
```

### Filtered Saw Wave
```yaml
midi-filtered:
  midi:
    channel: 0
    signal:
      filter:
        type: lowpass
        cutoff: 2000
        signal:
          osc:
            type: saw
            duration: 2.0
            amp: 0.6
            attack: 0.05
            release: 0.5
```

### Multi-Channel Setup
Create multiple MIDI instruments on different channels:

```yaml
midi-bass:
  midi:
    channel: 0
    signal:
      osc:
        type: sine
        duration: 0.5
        amp: 0.7

midi-lead:
  midi:
    channel: 1
    signal:
      osc:
        type: saw
        duration: 1.0
        amp: 0.5
```

## Usage

**Important:** Enable real-time mode in `config.py`:
```python
DO_PLAY_IN_REAL_TIME = True
```

Then run:
```bash
python waves.py midi-synth
```

The MIDI node runs indefinitely, listening for MIDI events until you stop it (Ctrl+C).

## Technical Details

### Frequency Conversion
MIDI notes → frequency using standard tuning (A4 = 440Hz):
```python
frequency = 440.0 * (2.0 ** ((note_number - 69) / 12.0))
```

### Polyphony
Unlimited polyphonic voices. Each note-on creates a new signal instance.

### Note Duration
- Uses the signal's `duration` parameter (e.g., 1.0 or 2.0 seconds)
- Default fallback: 2.0 seconds if not specified
- Note-off events are logged but don't interrupt playback (notes play to completion)

### Thread Safety
MIDI messages arrive on a separate thread via `mido` callback and are queued for safe processing in the audio render thread.

### Debug Mode
Enable MIDI debugging in `nodes/midi.py`:
```python
MIDI_DEBUG = True  # Shows note on/off messages
```

## Roadmap & Limitations

**Current Limitations:**
- Note-off doesn't trigger envelope release
- No MIDI CC support
- First available port only (no selection)
- No pitch bend or aftertouch

**Future Enhancements:**
- MIDI CC → parameter mapping
- Note-off envelope release
- Port selection parameter
- Pitch bend support
- MIDI learn functionality

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **No MIDI ports found** | Virtual port "waves_virtual" created automatically. Connect from DAW or create IAC bus in Audio MIDI Setup (macOS) |
| **Notes not triggering** | Check MIDI controller connection, verify correct channel, enable `MIDI_DEBUG = True` to see messages |
| **App stops after 2 seconds** | Enable real-time mode: `DO_PLAY_IN_REAL_TIME = True` in `config.py` |
| **Notes sound wrong** | Check signal envelope (attack/release), verify no frequency overrides, check for clipping |
| **Port already in use** | Close other applications using MIDI port |

## Dependencies

Required packages (in `requirements.txt`):
- `mido==1.3.3` - MIDI message handling
- `python-rtmidi==1.5.8` - Real-time MIDI I/O
