from enum import Enum


class RenderArgs(str, Enum):
    FREQUENCY = "frequency"
    FREQUENCY_MULTIPLIER = "frequency_multiplier"
    AMPLITUDE_MULTIPLIER = "amplitude_multiplier"
    DURATION = "duration"
