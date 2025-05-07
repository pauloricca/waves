from __future__ import annotations
from enum import Enum
from typing import Union, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, RootModel, field_validator, model_validator

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


class OscillatorModel(BaseModel):
    model_config = ConfigDict(extra='forbid')
    freq: Optional[WavableValue] = None
    freq_interpolation: InterpolationTypes = InterpolationTypes.LINEAR
    amp: WavableValue = 1.0
    amp_interpolation: InterpolationTypes = InterpolationTypes.LINEAR
    duration: Optional[float] = None
    attack: float = 0
    release: float = 0
    osc: OscillatorTypes = OscillatorTypes.SIN
    partials: List[OscillatorModel] = []
    scale: float = 1.0 # Perlin noise scale
    seed: float = 0.0 # Perlin noise seed
    min: Optional[float] = None # normalized min value
    max: Optional[float] = None # normalized max value
    
    @field_validator("osc", mode="before")
    @classmethod
    def normalize_wave_type(cls, v):
        if v is None:
            return OscillatorTypes.SIN.value
        if isinstance(v, str):
            return v.upper()
        return v


WavableValue = Union[float, List[Union[float, List[float]]], OscillatorModel]


class SequenceModel(BaseModel):
    model_config = ConfigDict(extra='forbid')
    interval: float = 0
    sequence: Optional[List[Union[str, List[str], None]]] = None
    chain: Optional[List[str]] = None


class SoundLibraryModel(RootModel[Dict[str, Union[OscillatorModel, SequenceModel]]]):
    @model_validator(mode="before")
    @classmethod
    def discriminate(cls, data):
        parsed = {}
        for k, v in data.items():
            if isinstance(v, dict) and ("sequence" in v or "chain" in v):
                parsed[k] = SequenceModel(**v)
            elif isinstance(v, dict) and ("freq" in v or "osc" in v):
                parsed[k] = OscillatorModel(**v)
        return parsed

    def __getitem__(self, key):
        return self.root[key]

    def keys(self):
        return self.root.keys()

