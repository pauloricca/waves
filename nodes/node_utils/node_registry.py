from nodes.delay import DELAY_DEFINITION
from nodes.non_real_time.envelope import ENVELOPE_DEFINITION
from nodes.filter import FILTER_DEFINITION
from nodes.deprecated.invert import INVERT_DEFINITION
from nodes.normalise import NORMALISE_DEFINITION
from nodes.oscillator import OSCILLATOR_DEFINITION
from nodes.sample import SAMPLE_DEFINITION
from nodes.non_real_time.sequencer import SEQUENCER_DEFINITION
from nodes.non_real_time.shuffle import SHUFFLE_DEFINITION
from nodes.smooth import SMOOTH_DEFINITION

NODE_REGISTRY = [
    OSCILLATOR_DEFINITION,
    SEQUENCER_DEFINITION,
    DELAY_DEFINITION,
    FILTER_DEFINITION,
    SHUFFLE_DEFINITION,
    NORMALISE_DEFINITION,
    SMOOTH_DEFINITION,
    INVERT_DEFINITION,
    SAMPLE_DEFINITION,
    ENVELOPE_DEFINITION,
]
