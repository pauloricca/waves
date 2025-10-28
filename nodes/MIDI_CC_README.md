# MIDI CC Node Documentation

## Overview

The `midi_cc` node captures MIDI Control Change (CC) messages and outputs them as a continuous wave/signal that can be used to modulate parameters of other nodes in real-time.

**Key Features:**
- ✅ Real-time CC capture - Responds instantly to MIDI controller movements
- ✅ Channel filtering - Listen to specific MIDI channels (0-15)
- ✅ CC number selection - Target specific controls (0-127)
- ✅ Initial value support - Define starting value before any CC is received
- ✅ Normalized output - CC values (0-127) converted to normalized range (0.0-1.0)
- ✅ Continuous operation - Runs indefinitely, updating as CC messages arrive

## Parameters

- **channel** (int): MIDI channel to listen to (0-15). Default: 0
- **cc** (WavableValue): The CC number to capture (0-127). Can be dynamic. Default: 0
- **initial** (float): The value in the output range before any CC is received. Default: 0.5
- **range** (Tuple[WavableValue, WavableValue]): Output range [min, max]. Both can be dynamic. Default: (0.0, 1.0)
- **device** (str | None): Optional device key from config.py. If None, uses the default device. Default: None

## Multi-Device Support

You can use MIDI CC from multiple controllers simultaneously. Configure devices in `config.py`:

**config.py:**
```python
MIDI_INPUT_DEVICES = {
    "main": "Arturia KeyStep 37",
    "pads": "AKAI MPD218",
}
MIDI_DEFAULT_DEVICE_KEY = "main"
```

**waves.yaml:**
```yaml
multi_device_cc:
  mix:
    signals:
      # Filter cutoff from main controller
      - filter:
          type: lowpass
          cutoff:
            midi_cc:
              channel: 0
              cc: 74
              range: [200, 5000]
              # device not specified = uses "main" (default)
          signal:
            osc:
              freq: 220
      
      # Amplitude from pads controller
      - osc:
          freq: 440
          amp:
            midi_cc:
              channel: 0
              cc: 1
              device: pads  # Explicitly use pads
              range: [0, 0.5]
```

**MIDI CC State Persistence:**
- CC values are saved to `.midi_state.json` every 2 seconds (configurable)
- Values are stored per device name (not device key)
- When you switch devices, each device's CC values are preserved separately
- This prevents confusion when reconnecting controllers

## How It Works

1. The node connects to the shared MIDI input (same as the `midi` node)
2. It monitors incoming MIDI Control Change messages
3. When a CC message matching the specified channel and CC number is received:
   - The CC value (0-127) is normalized to (0.0-1.0)
   - The current value is updated
4. The `render` method outputs a wave filled with the current value
5. The node runs indefinitely, continuously outputting the latest CC value

## Common CC Numbers

Here are some commonly used CC numbers:
- **1**: Modulation Wheel
- **7**: Volume
- **10**: Pan
- **11**: Expression
- **64**: Sustain Pedal (0-63 = off, 64-127 = on)
- **71**: Resonance/Timbre
- **74**: Brightness/Cutoff

## Configuration Examples

### Basic CC to Filter Cutoff
Control a lowpass filter cutoff with a MIDI knob (CC 74):

```yaml
midi-cc-filter:
  filter:
    type: lowpass
    cutoff:
      normalise:
        target_min: 200
        target_max: 5000
        signal:
          midi_cc:
            channel: 0
            cc_number: 74
            initial_value: 0.5  # Start at mid-point
    signal:
      osc:
        type: saw
        freq: 220
        amp: 0.5
```

### Modulation Wheel to Vibrato
Use the modulation wheel (CC 1) to control vibrato depth:

```yaml
midi-mod-vibrato:
  osc:
    type: sine
    freq: 440
    freq_mod:
      osc:
        type: sine
        freq: 5
        amp:
          normalise:
            target_min: 0
            target_max: 20  # Max vibrato of ±20 Hz
            signal:
              midi_cc:
                channel: 0
                cc_number: 1
                initial_value: 0.0  # No vibrato initially
```

### Volume Control
Use CC 7 (Volume) to control amplitude:

```yaml
midi-volume:
  osc:
    type: sine
    freq: 440
    amp:
      midi_cc:
        channel: 0
        cc_number: 7
        initial_value: 0.5  # Start at half volume
```

### Multiple CC Controls
Combine multiple CC inputs to control different parameters:

```yaml
midi-multi-cc:
  filter:
    type: lowpass
    cutoff:
      normalise:
        target_min: 200
        target_max: 8000
        signal:
          midi_cc:
            channel: 0
            cc_number: 74  # Brightness/Cutoff
            initial_value: 0.5
    resonance:
      normalise:
        target_min: 0.5
        target_max: 10
        signal:
          midi_cc:
            channel: 0
            cc_number: 71  # Resonance
            initial_value: 0.1
    signal:
      osc:
        type: saw
        freq: 220
        amp:
          midi_cc:
            channel: 0
            cc_number: 7  # Volume
            initial_value: 0.5
```

## Usage with MIDI Node

The `midi_cc` node works great alongside the `midi` node for comprehensive MIDI control:

```yaml
midi-complete:
  midi:
    channel: 0
    signal:
      filter:
        type: lowpass
        cutoff:
          normalise:
            target_min: 500
            target_max: 5000
            signal:
              midi_cc:
                channel: 0
                cc_number: 74
                initial_value: 0.7
        signal:
          osc:
            type: saw
            amp: 0.5
            attack: 0.01
            release: 0.3
```

In this example:
- MIDI notes control pitch (via the `midi` node)
- A MIDI CC controls the filter cutoff (via the `midi_cc` node)

## MIDI Setup

The `midi_cc` node uses the same MIDI input manager as the `midi` node, so the MIDI setup is identical. See [MIDI_README.md](MIDI_README.md) for details on MIDI port configuration.

## Tips

2. **Smoothing**: Consider using the `smooth` node if CC changes are too abrupt
3. **Initial Values**: Set appropriate initial values so parameters have sensible defaults before you move the controller
4. **Testing**: Use `midi_testing.py` to identify available MIDI controllers and their CC outputs
