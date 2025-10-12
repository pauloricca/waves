YAML_FILE = "waves.yaml"
OUTPUT_DIR = "output"
OSC_ENVELOPE_TYPE = "exponential"  # Options: "linear", "exponential"
SAMPLE_RATE = 44100
BUFFER_SIZE = 2048

# Node reference settings
MAX_RECURSION_DEPTH = 3  # Maximum recursion depth for feedback loops

# MIDI settings
MIDI_INPUT_DEVICE_NAME = None  # Set to a specific device name, or None to auto-detect
VISUALISATION_ROW_HEIGHT = 10
RENDERED_MASTER_GAIN = 0.2
DO_NORMALISE_EACH_SOUND = False
DEFAULT_PLAYBACK_TIME = 4  # seconds, for nodes without explicit duration

WAIT_FOR_CHANGES_IN_WAVES_YAML = False

# Playback settings
DO_PRE_RENDER_WHOLE_SOUND = False
DO_PLAY_IN_REAL_TIME = True

# Visualisation settings
DO_VISUALISE_OUTPUT = True
DISPLAY_RENDER_TIME_PERCENTAGE = True
DO_ONLY_VISUALISE_ONE_BUFFER = False
VISUALISATION_FPS = 25  # Lower FPS reduces CPU usage and audio interference

DISABLE_GARBAGE_COLLECTION = False
