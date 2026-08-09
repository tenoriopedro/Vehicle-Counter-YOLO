from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ultralytics import YOLO
from ultralytics.engine.results import Results

from vehicle_counter.domain.entities import DetectedObject, VehicleClass


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


def build_yolo_stream(
    model_weights: Path, video_source: Path, classes_to_count: list[int]
) -> Iterable[list[DetectedObject]]:
    """
    Initializes the YOLO model and returns a lazy generator of domain entities.
    """
    model_yolo = YOLO(model_weights)

    # Results is an iterator, preserving the lazy evaluation
    results = model_yolo.track(  # type: ignore
        source=str(video_source),
        stream=True,
        persist=True,
        classes=classes_to_count,
        conf=0.5,
        iou=0.45,
        imgsz=608,
    )

    return transfer_yolo(results)
