# Real-time Recording Feature

## Overview

The real-time recording feature allows you to capture audio rendered during real-time playback without impacting performance. The recording is saved automatically when playback stops or when you press Ctrl-C.

## Configuration

Add these settings to `config.py`:

```python
# Real-time recording settings
DO_RECORD_REAL_TIME = False  # Enable to save real-time playback to file
REAL_TIME_RECORDING_FILENAME = "realtime_recording.wav"  # Output filename in OUTPUT_DIR
```

## Usage

1. **Enable recording** in `config.py`:
   ```python
   DO_RECORD_REAL_TIME = True
   ```

2. **Run your sound** as usual:
   ```bash
   ./waves.py seq
   ```

3. **Stop playback** in one of two ways:
   - Let it finish naturally (if duration is specified)
   - Press **Ctrl-C** to stop and save

4. **Find your recording** in the `output/` directory with the filename specified in config.

## Performance

The implementation is designed to be highly performant:

- **No blocking**: Audio samples are added to a `deque` (double-ended queue) which is O(1) for append operations
- **Thread-safe**: Uses Python's native thread-safe collections
- **Minimal overhead**: Only extends a buffer, no I/O during playback
- **Deferred write**: File is written only when playback stops, not during rendering
- **Signal handling**: Graceful Ctrl-C handling ensures recording is always saved

## Technical Details

- Recording captures audio **after master gain is applied** but **before clipping**
- Uses `atexit` for normal termination and `signal.SIGINT` for Ctrl-C
- Recording buffer is a global `deque` that persists across chunks
- Only works in real-time playback mode (`DO_PLAY_IN_REAL_TIME = True`)

## Example

```python
# config.py
DO_PLAY_IN_REAL_TIME = True
DO_RECORD_REAL_TIME = True
REAL_TIME_RECORDING_FILENAME = "my_jam_session.wav"
```

Then:
```bash
./waves.py my_sound
# Play around with MIDI controller or let it play
# Press Ctrl-C when done
# Recording saved: 23.45 seconds
```
