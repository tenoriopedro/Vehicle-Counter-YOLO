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

    @staticmethod
    def _ccw(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> bool:
        """
        Evaluates if three points are listed in a counter-clockwise order.
        """
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    @classmethod
    def _segments_intersect(
        cls,
        a: tuple[int, int],
        b: tuple[int, int],
        c: tuple[int, int],
        d: tuple[int, int],
    ) -> bool:
        """
        Returns True if line segment AB intersects line segment CD.
        """
        return cls._ccw(a, c, d) != cls._ccw(b, c, d) and cls._ccw(a, b, c) != cls._ccw(
            a, b, d
        )

    def _register_crossings(
        self, detected_objects: list[DetectedObject], current_time: float
    ) -> None:

        for obj in detected_objects:
            _x1, _y1, _x2, _y2 = obj.xyxy

            # Calculate bottom-center coordinate for intersection accuracy
            x_center, y_bottom = int((_x1 + _x2) / 2), int(_y2)
            current_point = (x_center, y_bottom)

            self.last_seen_timestamp[obj.id] = current_time

            # Retrieve T - 1 spatial state to establish movement vector
            previous_point = self.track_history.get(obj.id)
            self.track_history[obj.id] = current_point

            direction_val = self._check_intersection_point(
                previous_point, current_point
            )

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
        self, previous_point: tuple[int, int] | None, current_point: tuple[int, int]
    ) -> int | None:
        """
        Determines if a movement vector crossed the calibration line and its direction.
        Returns 0 for South, 1 for North, or None if no crossing occurred.
        """

        if previous_point is None:
            return None

        # Segment A: Calibration line (Static Anchor)
        line_a = self.line_points[0]
        line_b = self.line_points[1]

        # Segment B: Vehicle movement vector (T-1 to T)
        vec_c = previous_point
        vec_d = current_point

        # Absolute mathematical proof of physical crossing
        if self._segments_intersect(line_a, line_b, vec_c, vec_d):
            _, y_old = previous_point
            _, y_new = current_point

            # Logical direction derived strictly from the movement vector Y-axis delta
            if y_new > y_old:
                return 0  # Southbound
            if y_new < y_old:
                return 1  # Northbound

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
