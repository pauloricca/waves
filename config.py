SOUNDS_DIR = "sounds"  # Directory containing YAML sound definition files
OUTPUT_DIR = "output"
OSC_ENVELOPE_TYPE = "exponential"  # Options: "linear", "exponential"
SAMPLE_RATE = 44100
BUFFER_SIZE = 2048

# Node reference settings
MAX_RECURSION_DEPTH = 2  # Maximum recursion depth for feedback loops

# MIDI settings
# Dictionary mapping device keys to device names
MIDI_INPUT_DEVICES = {
    "korg": "nanoKONTROL2 SLIDER/KNOB",
    "akai": "LPD8",
}
MIDI_DEFAULT_DEVICE_KEY = 'korg'  # Default device key to use when not specified in YAML, or None to auto-detect
DO_PERSIST_MIDI_CC_VALUES = True  # Save MIDI CC values to file for persistence across restarts
MIDI_CC_SAVE_INTERVAL = 2.0  # Seconds between saving CC values to file


VISUALISATION_ROW_HEIGHT = 10
RENDERED_MASTER_GAIN = 0.6
DO_NORMALISE_EACH_SOUND = False
DEFAULT_PLAYBACK_TIME = 4  # seconds, for nodes without explicit duration


# Hot reload settings
DO_HOT_RELOAD = True
HOT_RELOAD_DELAY = 0.1  # Seconds to wait before starting reload (gives audio time to stabilize)

# Playback settings
DO_PRE_RENDER_WHOLE_SOUND = False
DO_PLAY_IN_REAL_TIME = True

# Real-time recording settings
DO_RECORD_REAL_TIME = False  # Enable to save real-time playback to file

# Visualisation settings
DO_VISUALISE_OUTPUT = False
DISPLAY_RENDER_STATS = True
DO_ONLY_VISUALISE_ONE_BUFFER = False
VISUALISATION_FPS = 20  # Lower FPS reduces CPU usage and audio interference

DISABLE_GARBAGE_COLLECTION = False
