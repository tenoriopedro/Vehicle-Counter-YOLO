from collections.abc import Iterable
from datetime import UTC, datetime, timedelta

from vehicle_counter.domain.entities import DetectedObject
from vehicle_counter.domain.events import Direction, VehicleEvent
from vehicle_counter.io.sink import TelemetrySink


class TrafficCounter:
    """
    Core domain service responsible for analyzing spatial coordinates over time
    to determine if vehicles cross a calibrated counting line.
    Stateful analyzer, but memory footprint is managed via TTL cleanup.
    """

    def __init__(
        self,
        frames_of_objects: Iterable[list[DetectedObject]],
        fps: float,
        sink: TelemetrySink,
        line_points: list[tuple[int, int]],
    ) -> None:

        self.frames_of_objects = frames_of_objects
        self.line_points = line_points
        self.fps = fps
        self.sink = sink

        self.processed_ids: set[int] = set()
        self.track_history: dict[int, tuple[int, int]] = {}
        self.last_seen_timestamp: dict[int, float] = {}
        self.ttl_seconds: float = 2.0

        self.start_time = datetime.now(UTC)

    def start_tracking(self) -> None:
        """
        Consumes the stream of detected objects, registering crossings and
        managing the temporal state of active tracks.
        """

        for frame_counter, detected_objects in enumerate(self.frames_of_objects):
            current_time = round(frame_counter / self.fps, 2)

            self._register_crossings(detected_objects, current_time)
            self._cleanup_memory(current_time)

    def _register_crossings(
        self, detected_objects: list[DetectedObject], current_time: float
    ) -> None:

        for obj in detected_objects:
            _x1, _y1, _x2, _y2 = obj.xyxy

            # Calculate bottom-center coordinate for intersection accuracy
            x_center, y_bottom = int((_x1 + _x2) / 2), int(_y2)

            self.last_seen_timestamp[obj.id] = current_time

            # Retrieve T - 1 spatial state to establish movement vector
            previous_point = self.track_history.get(obj.id)
            self.track_history[obj.id] = (x_center, y_bottom)

            direction_val = self._check_intersection_point(previous_point, y_bottom)

            if direction_val is not None and obj.id not in self.processed_ids:
                self.processed_ids.add(obj.id)

                enum_direction = Direction(direction_val)

                timestamp: datetime = self.start_time + timedelta(seconds=current_time)

                event = VehicleEvent(
                    timestamp=timestamp,
                    track_id=obj.id,
                    class_id=obj.cls_id,
                    confidence=obj.conf,
                    direction=enum_direction,
                )

                self.sink.add(event)

    def _check_intersection_point(
        self, previous_point: tuple[int, int] | None, y_bottom: int
    ) -> int | None:
        """
        Determines if a movement vector crossed the calibration line and its direction.
        Returns 0 for South, 1 for North, or None if no crossing occurred.
        """

        if previous_point is None:
            return None

        intersection_point = self.line_points[0][1]

        _, _y_old = previous_point

        # Southbound logic
        if _y_old < intersection_point and y_bottom >= intersection_point:
            return 0

        # Northbound logic
        if _y_old > intersection_point and y_bottom <= intersection_point:
            return 1

        return None

    def _cleanup_memory(self, current_timestamp: float) -> None:
        """
        Purges stale tracking data from memory to prevent infinite RAM growth
        during prolonged video processing sessions.
        """

        keys_to_delete = [
            key
            for key, last_seen in self.last_seen_timestamp.items()
            if (current_timestamp - last_seen) > self.ttl_seconds
        ]

        for key in keys_to_delete:
            self.last_seen_timestamp.pop(key, None)
            self.track_history.pop(key, None)
            self.processed_ids.discard(key)


if __name__ == "__main__":
    ...
