from nodes.delay import DELAY_DEFINITION
from nodes.envelope import ENVELOPE_DEFINITION
from nodes.filter import FILTER_DEFINITION
from nodes.deprecated.invert import INVERT_DEFINITION
from nodes.map import MAP_DEFINITION
from nodes.midi import MIDI_DEFINITION
from nodes.midi_cc import MIDI_CC_DEFINITION
from nodes.mix import MIX_DEFINITION
from nodes.oscillator import OSCILLATOR_DEFINITION
from nodes.reference import REFERENCE_DEFINITION
from nodes.sample import SAMPLE_DEFINITION
from nodes.sequencer import SEQUENCER_DEFINITION
from nodes.shuffle import SHUFFLE_DEFINITION
from nodes.smooth import SMOOTH_DEFINITION

NODE_REGISTRY = [
    OSCILLATOR_DEFINITION,
    SEQUENCER_DEFINITION,
    DELAY_DEFINITION,
    FILTER_DEFINITION,
    SHUFFLE_DEFINITION,
    MAP_DEFINITION,
    SMOOTH_DEFINITION,
    INVERT_DEFINITION,
    SAMPLE_DEFINITION,
    ENVELOPE_DEFINITION,
    MIDI_DEFINITION,
    MIDI_CC_DEFINITION,
    MIX_DEFINITION,
    REFERENCE_DEFINITION,
]
