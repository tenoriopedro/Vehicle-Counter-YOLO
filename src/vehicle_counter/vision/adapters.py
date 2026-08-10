from collections.abc import Iterable
from pathlib import Path
from typing import Any

from cv2.typing import MatLike
from ultralytics import YOLO
from ultralytics.engine.results import Results

from vehicle_counter.domain.entities import DetectedObject, VehicleClass
from vehicle_counter.vision.streamer import VideoStreamer


def transfer_yolo(yolo_results: list[Results]) -> Iterable[list[DetectedObject]]:
    """
    Acts as an anti-corruption layer for the computer vision models.
    Converts raw YOLO tensor outputs into pure Python domain entities (DetectedObject).
    Yields data lazily frame-by-frame to minimize memory footprint.

    Args:
        yolo_results: An iterable of YOLO Result objects.

    Yields:
        A list of strongly typed DetectedObject entities for a single frame.
    """

    for result in yolo_results:
        detected_objects: list[DetectedObject] = []

        # YOLO returns generic objects that bypass standard typing.
        raw_boxes: Any = result.boxes

        if raw_boxes is None or raw_boxes.id is None:
            yield detected_objects
            continue

        for box in raw_boxes:
            if box.id is None:
                continue

            raw_cls_id = int(box.cls[0].item())
            try:
                cls_id = VehicleClass(raw_cls_id)
            except ValueError:
                # Ignores detections that do not map to the
                # system's allowed vehicle classes
                continue

            track_id = int(box.id[0].item())
            conf = round(float(box.conf[0].item()), 2)

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            coords = (float(x1), float(y1), float(x2), float(y2))

            obj = DetectedObject(id=track_id, xyxy=coords, cls_id=cls_id, conf=conf)

            detected_objects.append(obj)

        yield detected_objects


def frame_generator(streamer: VideoStreamer) -> Iterable[MatLike]:
    """
    Lazy generator consuming frames from RAM.
    Ensures graceful termination when the producer signals the end.
    """
    while True:
        frame = streamer.read()
        if frame is None:
            break

        # Ignores dummy frames created due to disk latency
        if frame.shape == (1, 1, 3):
            continue

        yield frame


def build_yolo_stream(
    model_weights: Path, video_source: Path, classes_to_count: list[int]
) -> Iterable[list[DetectedObject]]:
    """
    Initializes the YOLO model and orchestrates the Producer-Consumer pattern.
    """
    model_yolo = YOLO(model_weights, task="detect")

    # Start the I/O disk thread
    streamer = VideoStreamer(str(video_source), queue_size=60).start()

    try:
        # YOLO consumes purely from RAM memory
        # type: ignore is required because Ultralytics type stubs lack Iterable support
        results = model_yolo.track(  # type: ignore
            source=frame_generator(streamer),  # type: ignore
            stream=True,
            persist=True,
            classes=classes_to_count,
            conf=0.5,
            iou=0.45,
            imgsz=608,
        )

        yield from transfer_yolo(results)

    finally:
        # Ensures the producer thread is killed even if YOLO crashes
        streamer.stop()
