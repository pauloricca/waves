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

# MIDI output settings
MIDI_OUTPUT_DEVICE = None  # MIDI output device name, or None to use first available (for MIDI clock out)
MIDI_CLOCK_ENABLED = True  # Enable MIDI clock output when tempo node has source='internal'


RENDERED_MASTER_GAIN = 1
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
DO_SAVE_MULTITRACK = True  # Save individual track stems when using tracks node (applies to both realtime and non-realtime)

# Visualisation settings
DO_VISUALISE_OUTPUT = True
DISPLAY_RENDER_STATS = True
DO_ONLY_VISUALISE_ONE_BUFFER = False
DO_SCRAMBLE_VISUALISATION_ROWS = False  # Scramble row order for interesting glitch effect
VISUALISATION_ROW_HEIGHT = 10
VISUALISATION_FPS = 30
CHANCE_OF_CLEARING_ONE_LESS_ROW = 0  # Chance of clearing one row each frame for glitch effect

DISABLE_GARBAGE_COLLECTION = False
