from __future__ import annotations
from enum import Enum
from typing import Union, Dict, List, Optional
from pydantic import BaseModel, RootModel, field_validator, model_validator

class WaveTypes(str, Enum):
    SIN = "SIN"
    COS = "COS"
    TRI = "TRI"
    SQR = "SQR"
    SAW = "SAW"
    NOISE = "NOISE"
    NONE = "NONE"


class InterpolationTypes(str, Enum):
    LINEAR = "LINEAR"
    SMOOTH = "SMOOTH"
    STEP = "STEP"


class WaveModel(BaseModel):
    freq: Optional[WavableValue] = None
    freq_interpolation: InterpolationTypes = InterpolationTypes.LINEAR
    amp: WavableValue = 1.0
    amp_interpolation: InterpolationTypes = InterpolationTypes.LINEAR
    duration: Optional[float] = None
    attack: float = 0
    release: float = 0
    type: WaveTypes = WaveTypes.SIN
    partials: List[WaveModel] = []
    filters: List[dict] = []
    min: Optional[float] = None
    max: Optional[float] = None
    
    @field_validator("type", mode="before")
    @classmethod
    def normalize_wave_type(cls, v):
        if v is None:
            return WaveTypes.SIN.value
        if isinstance(v, str):
            return v.upper()
        return v


WavableValue = Union[float, List[Union[float, List[float]]], WaveModel]


class SequenceModel(BaseModel):
    interval: float = 0
    sequence: Optional[List[Union[str, List[str], None]]] = None
    chain: Optional[List[str]] = None


class SoundLibraryModel(RootModel[Dict[str, Union[WaveModel, SequenceModel]]]):
    @model_validator(mode="before")
    @classmethod
    def discriminate(cls, data):
        parsed = {}
        for k, v in data.items():
            if isinstance(v, dict) and ("sequence" in v or "chain" in v):
                parsed[k] = SequenceModel(**v)
            else:
                parsed[k] = WaveModel(**v)
        return parsed

    def __getitem__(self, key):
        return self.root[key]

    def keys(self):
        return self.root.keys()

