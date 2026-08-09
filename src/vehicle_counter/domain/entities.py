from dataclasses import dataclass
from enum import IntEnum


class Direction(IntEnum):
    """Represents the strictly allowed movement directions in the system."""

    SOUTH = 0
    NORTH = 1

    @property
    def label(self) -> str:
        match self:
            case Direction.SOUTH:
                return "South"
            case Direction.NORTH:
                return "North"


class VehicleClass(IntEnum):
    """Represents the strictly allowed vehicle categories in the system."""

    CAR = 2
    MOTORCYCLE = 3
    BUS = 5
    TRUCK = 7

    @property
    def label(self) -> str:
        match self:
            case VehicleClass.CAR:
                return "Cars"
            case VehicleClass.MOTORCYCLE:
                return "Motorcycles"
            case VehicleClass.BUS:
                return "Buses"
            case VehicleClass.TRUCK:
                return "Trucks"


@dataclass(frozen=True)
class DetectedObject:
    """Immutable data transfer object representing a single detection frame."""

    id: int
    xyxy: tuple[float, float, float, float]
    cls_id: VehicleClass
    conf: float


@dataclass(frozen=True)
class TrafficReportDTO:
    """Immutable contract for transporting aggregated
    traffic data to presentation layers."""

    valid_classes_counts: dict[VehicleClass, int]
    valid_directions_counts: dict[Direction, int]
    total_anomalies: int
