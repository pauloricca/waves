from nodes.automation import AUTOMATION_DEFINITION
from nodes.context import CONTEXT_DEFINITION
from nodes.delay import DELAY_DEFINITION
from nodes.retrigger import RETRIGGER_DEFINITION
from nodes.envelope import ENVELOPE_DEFINITION
from nodes.expression import EXPRESSION_DEFINITION
from nodes.filter import FILTER_DEFINITION
from nodes.follow import FOLLOW_DEFINITION
from nodes.hold import HOLD_DEFINITION
from nodes.input import INPUT_DEFINITION
from nodes.non_realtime.invert import INVERT_DEFINITION
from nodes.map import MAP_DEFINITION
from nodes.midi import MIDI_DEFINITION
from nodes.midi_cc import MIDI_CC_DEFINITION
from nodes.mix import MIX_DEFINITION
from nodes.oscillator import OSCILLATOR_DEFINITION
from nodes.reference import REFERENCE_DEFINITION
from nodes.sample import SAMPLE_DEFINITION
from nodes.select import SELECT_DEFINITION
from nodes.sequencer import SEQUENCER_DEFINITION
from nodes.shuffle import SHUFFLE_DEFINITION
from nodes.smooth import SMOOTH_DEFINITION
from nodes.snap import SNAP_DEFINITION
from nodes.tempo import TEMPO_DEFINITION

NODE_REGISTRY = [
    AUTOMATION_DEFINITION,
    OSCILLATOR_DEFINITION,
    SEQUENCER_DEFINITION,
    DELAY_DEFINITION,
    RETRIGGER_DEFINITION,
    FILTER_DEFINITION,
    SHUFFLE_DEFINITION,
    MAP_DEFINITION,
    SMOOTH_DEFINITION,
    SNAP_DEFINITION,
    SELECT_DEFINITION,
    TEMPO_DEFINITION,
    INVERT_DEFINITION,
    SAMPLE_DEFINITION,
    ENVELOPE_DEFINITION,
    EXPRESSION_DEFINITION,
    CONTEXT_DEFINITION,
    MIDI_DEFINITION,
    MIDI_CC_DEFINITION,
    MIX_DEFINITION,
    REFERENCE_DEFINITION,
    HOLD_DEFINITION,
    FOLLOW_DEFINITION,
    INPUT_DEFINITION,
]
