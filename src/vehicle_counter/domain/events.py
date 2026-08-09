from datetime import UTC, datetime

from pydantic import BaseModel, Field, StrictFloat, StrictInt

from vehicle_counter.domain.entities import Direction, VehicleClass


class VehicleEvent(BaseModel):
    """
    Schema for validating streaming vehicle tracking events.
    Ensures strict type compliance before serialization to storage.
    """

    track_id: StrictInt
    class_id: VehicleClass
    confidence: StrictFloat = Field(ge=0.0, le=1.0)
    direction: Direction
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
