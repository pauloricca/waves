from __future__ import annotations
from enum import Enum
from typing import Union, List, Optional
from pydantic import BaseModel, ConfigDict, field_validator

class OscillatorTypes(str, Enum):
    SIN = "SIN"
    COS = "COS"
    TRI = "TRI"
    SQR = "SQR"
    SAW = "SAW"
    NOISE = "NOISE"
    PERLIN = "PERLIN"
    NONE = "NONE"


class InterpolationTypes(str, Enum):
    LINEAR = "LINEAR"
    SMOOTH = "SMOOTH"
    STEP = "STEP"


class BaseNodeModel(BaseModel):
    pass


class OscillatorModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    type: OscillatorTypes = OscillatorTypes.SIN
    freq: Optional[WavableValue] = None
    freq_interpolation: InterpolationTypes = InterpolationTypes.LINEAR
    amp: WavableValue = 1.0
    amp_interpolation: InterpolationTypes = InterpolationTypes.LINEAR
    duration: Optional[float] = None
    attack: float = 0
    release: float = 0
    partials: List[OscillatorModel] = []
    scale: float = 1.0 # Perlin noise scale
    seed: Optional[float] = None # Perlin noise seed
    min: Optional[float] = None # normalized min value
    max: Optional[float] = None # normalized max value
    
    @field_validator("type", mode="before")
    @classmethod
    def normalize_wave_type(cls, v):
        if v is None:
            return OscillatorTypes.SIN.value
        if isinstance(v, str):
            return v.upper()
        return v


WavableValue = Union[float, List[Union[float, List[float]]], BaseNodeModel]


class SequencerModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    interval: float = 0
    repeat: int = 1
    sequence: Optional[List[Union[str, List[str], None]]] = None
    chain: Optional[List[str]] = None


class DelayModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    time: float = 0.1
    repeats: int = 3
    feedback: float = 0.3
    do_trim: bool = False
    signal: BaseNodeModel = None
