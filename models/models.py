from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


class BaseNodeModel(BaseModel):
    duration: Optional[float] = None
    pass
