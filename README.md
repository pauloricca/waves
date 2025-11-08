# Waves

A Python/NumPy-based experimental sound design system for creating sounds, sequences, and compositions through procedural modular synthesis. Define sounds in YAML and play them in realtime with hot-reload support for live editing.

## Overview

Waves is a flexible audio synthesis framework where sounds are created by connecting modules (called nodes) together. The system treats everything as waves - there's no distinction between audio and control signals, enabling creative modulation routing and sound design possibilities.

**Key Features:**
- **YAML-based sound definition**: Describe complex sounds using simple, readable YAML syntax
- **Modular architecture**: Connect oscillators, filters, envelopes, sequencers, and effects
- **Realtime playback with hot reload**: Edit YAML files and hear changes instantly without stopping playback
- **Expression system**: Use Python expressions for dynamic, mathematical sound design
- **Multi-file organization**: Organize sounds across multiple YAML files with cross-referencing
- **MIDI support**: Use MIDI controllers for live performance and parameter control
- **Stereo mixing with tracks**: Multi-track mixing with panning and individual stem export

## Installation

### Quick Install (Recommended)

```bash
git clone https://github.com/pauloricca/waves.git && cd waves && ./install
```

The install script will:
- Create a virtual environment
- Upgrade pip
- Install all dependencies from requirements.txt


## Quick Start

### Basic Usage

Run a sound defined in any YAML file in the `sounds/` directory:

```bash
# Using the convenience script
./play sound_name
```

**The `play` script** automatically activates the virtual environment if needed, so you don't have to remember to do it manually each time.

The system will:
1. Load all YAML files from `sounds/` directory
2. Play the specified sound
3. Watch for file changes and automatically reload

### Example 1: Simple Sine Wave

Create a file `sounds/my_sounds.yaml`:

```yaml
simple_tone:
  osc:
    type: sin
    freq: 440
    amp: 0.5
    duration: 2
```

Run it:
```bash
./play simple_tone
```

### Example 2: FM Synthesis with LFO

```yaml
fm_sound:
  osc:
    type: sin
    freq: 
      osc:
        type: sin
        freq: 5
        range: [400, 600]
    duration: 3
```

This creates an FM sound where a 5 Hz sine wave modulates the carrier frequency between 400-600 Hz.

### Example 3: Simple Beat Pattern

```yaml
beat:
  sequencer:
    interval: 0.5
    repeat: 8
    steps:
      - kick
      - hihat
      - snare
      - hihat

kick:
  osc:
    type: sin
    freq: [60, 40]
    duration: 0.2

hihat:
  osc:
    type: noise
    duration: 0.05

snare:
  mix:
    signals:
      - osc:
          type: sin
          freq: 200
          duration: 0.15
      - osc:
          type: noise
          duration: 0.1
```

### Example 4: Expression-Based Sound Design

```yaml
vars:
  bpm: 120
  root_note: 261.63  # Middle C

rising_tone:
  osc:
    type: sin
    freq: "root_note * (1 + t * 2)"  # Frequency rises over time
    amp: "0.5 * (1 - t / duration)"   # Amplitude falls over time
    duration: 2
```

## Configuration

Edit `config.py` to customize:

- **Sample rate**: `SAMPLE_RATE = 44100`
- **Buffer size**: `BUFFER_SIZE = 512`
- **Realtime vs non-realtime**: `DO_REALTIME_PLAYBACK = True`
- **Hot reload**: `DO_HOT_RELOAD = True`
- **Multi-track export**: `DO_SAVE_MULTITRACK = True`
- **Sounds directory**: `SOUNDS_DIR = "sounds"`
- **Output directory**: `OUTPUT_DIR = "output"`

## Available Node Types

### Generators
- **osc**: Oscillators (sin, tri, sqr, saw, noise)
- **sample**: Audio file playback with speed/pitch control
- **midi_in**: MIDI note input for live performance

### Processors
- **filter**: Filters (lowpass, highpass, bandpass, notch)
- **envelope**: ADSR envelopes
- **delay**: Delay effects with feedback
- **follow**: Envelope follower

### Modulators
- **automation**: Interpolated parameter automation
- **midi_cc**: MIDI CC input
- **expression**: Arbitrary Python expressions with multiple inputs

### Structure
- **mix**: Mix multiple signals together
- **sequencer**: Step sequencer for patterns
- **tracks**: Multi-track stereo mixer with panning
- **context**: Set variables/parameters for child nodes
- **tempo**: Set BPM for child nodes

### Utilities
- **reference**: Reference nodes by ID for advanced routing
- **map**: Map values from one range to another
- **smooth**: Smooth parameter changes
- **select**: Choose between signals based on condition

See individual node files in `nodes/` for detailed parameters and usage.

## MIDI Support

Connect a MIDI controller and use `midi_in` for note input or `midi_cc` for parameter control:

```yaml
midi_synth:
  midi_in:
    signal:
      osc:
        type: saw
        freq: $freq  # Uses MIDI note frequency
        amp: $gate   # Uses MIDI velocity
```

## Sub-Patching and Reuse

Define sounds once and reuse them with different parameters:

```yaml
# Define base sound
my_bass:
  osc:
    type: saw
    freq: 100
    duration: 1

# Use it multiple times with overrides
composition:
  mix:
    signals:
      - my_bass:
          freq: 100
      - my_bass:
          freq: 150
      - my_bass:
          freq: 200
```

## Output

Audio files are saved to the `output/` directory when:
- A sound finishes playing in non-realtime mode
- You stop playback (Ctrl+C) in realtime mode

Multi-track sounds export:
- Individual track stems: `{sound_name}__{track_name}.wav`
- Final mixdown: `{sound_name}.wav`

## Development

The project structure:
- `waves.py`: Main entry point
- `config.py`: Global configuration
- `nodes/`: All node implementations
- `sounds/`: YAML sound definitions
- `instructions/`: Detailed documentation on features
