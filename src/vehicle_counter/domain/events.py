from datetime import UTC, datetime
from enum import IntEnum

from pydantic import BaseModel, Field, StrictFloat, StrictInt


class Direction(IntEnum):
    SOUTH = 0
    NORTH = 1


class VehicleEvent(BaseModel):

    track_id: StrictInt
    class_id: StrictInt
    confidence: StrictFloat = Field(ge=0.0, le=1.0)
    direction: Direction

    timestamp : datetime = Field(default_factory=lambda: datetime.now(UTC))

