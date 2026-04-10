from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO
from ultralytics.engine.results import Boxes


class TrafficCounter:
    def __init__(
        self,
        model_path: Path,
        video_source: Path,
        output_dir: Path,
        classes_to_count: list[int],
        line_points: list[tuple[int, int]],
        conf: float = 0.1,
        file_name: str = "video_result") -> None:

        self.model_path = model_path
        self.video_source = video_source
        self.output_dir = output_dir
        self.class_to_count = classes_to_count
        self.line_points = line_points
        self.conf = conf
        self.file_name = file_name

        if not self.model_path.exists():
            msg = "Modelo não encontrado"
            raise FileNotFoundError(msg)

        if not self.video_source.exists():
            msg = "Video para compilação não encontrado"
            raise FileNotFoundError(msg)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        total_files = len(list(self.output_dir.glob("*.parquet")))
        self.batch_counter = total_files + 1

        # Initialize Model
        self.model = YOLO(self.model_path)

        self.detected_data = {
            "timestamp": [],
            "track_id": [],
            "class_id": [],
            "confidence": [],
            "direction": [],
        }

        self.processed_ids = set()

        self.track_history: dict = {}

        self.flush_limit = 10

    def _register_crossings(
            self,
            boxes: Boxes,
            current_time: float) -> None:

        """
        Extracts tracking data from YOLO boxes and handles vehicle state registration.

        Iterates through all detected objects, updates their spatial history, and
        records valid line crossings into the detected_data dictionary.

        Args:
            boxes (Boxes): Ultralytics Boxes object containing IDs and coordinates.
            current_time (float): The current timestamp in seconds.
    """

        if boxes.id is None:
            return

        for track_id, xyxy, cls_id, conf in zip(
            boxes.id, boxes.xyxy, boxes.cls, boxes.conf, strict=False):

            track_id = int(track_id)
            cls_id = int(cls_id)
            conf = round(float(conf), 2)

            _x1, _y1, _x2, _y2 = xyxy

            # Calculate bottom-center coordinate for intersection accuracy
            x_center, y_bottom = int((_x1 + _x2) / 2), int(_y2)

            # Retrieve T - 1 spatial state to establish movement vector
            previous_point = self.track_history.get(track_id)

            self.track_history[track_id] = (x_center, y_bottom)

            direction = self._check_intersection_point(previous_point, y_bottom)


            if direction is not None and \
            track_id not in self.processed_ids:
                self.processed_ids.add(track_id)

                self.detected_data["timestamp"].append(current_time)
                self.detected_data["track_id"].append(track_id)
                self.detected_data["class_id"].append(cls_id)
                self.detected_data["confidence"].append(conf)
                self.detected_data["direction"].append(direction)

    def _check_intersection_point(
            self,
            previous_point: tuple[int, int] | None,
            y_bottom: int) -> int | None:

        """
        Validates if a vehicle's movement vector has crossed the virtual line.

        Args:
            previous_point (tuple[int, int] | None): The (x, y) coordinates
            from the last frame.
            y_bottom (int): The current vertical base coordinate.

        Returns:
            int | None: 0 for South, 1 for North, or None if no crossing occurred.
    """

        if previous_point is None:
            return None

        intersection_point = self.line_points[0][1]

        _, _y_old = previous_point

        # South
        if _y_old < intersection_point and y_bottom >= intersection_point:

            return 0

        # North
        if _y_old > intersection_point and y_bottom <= intersection_point:

            return 1

        return None

    def _flush_to_disk(self) -> None:

        """
        Exports accumulated tracking data to disk and clears memory.

        Converts the in-memory dictionary to a Pandas DataFrame
        and saves it as a highly compressed Parquet file
        using the PyArrow engine.
        The filename is suffixed with a zero-padded batch number
        (e.g., _001) to ensure sequential ordering.
        After a successful export, all internal lists
        are cleared to free up RAM for the next batch.
        """

        if len(self.detected_data["timestamp"]) == 0:
            return

        df = pd.DataFrame(data=self.detected_data)

        counter = str(self.batch_counter)

        parquet_file = f"{self.file_name}_{counter.zfill(3)}.parquet"

        save_parquet_file = self.output_dir / parquet_file

        df.to_parquet(save_parquet_file, engine="pyarrow")

        self.batch_counter += 1

        for key in self.detected_data:
            self.detected_data[key].clear()

    def start_tracking(self) -> None:

        """
        Initiates the video capture and coordinates
        the frame-by-frame tracking pipeline.

        Raises:
            ValueError: If the video metadata (FPS) cannot be correctly extracted.
        """

        cap = cv2.VideoCapture(str(self.video_source))

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            msg = (
                "Falha ao extrair metadados do video. "
                f"Obtido: FPS={fps}\n"
                "Verifique a integridade da fonte de vídeo"
            )
            raise ValueError(msg)

        frame_counter = 0

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    print(
                        "Video frame is empty "
                        "or video processing has been successfully completed."
                    )
                    break

                results = self.model.track(
                    source=frame,
                    persist=True,
                    classes=self.class_to_count,
                    conf=self.conf
                )

                for result in results:
                    boxes = result.boxes.cpu().numpy()  # type: ignore

                    current_time = round(frame_counter / fps, 2)

                    self._register_crossings(boxes, current_time)

                    if len(self.detected_data["timestamp"]) >= self.flush_limit:
                        self._flush_to_disk()

                frame_counter += 1

        finally:
            self._flush_to_disk()


def run() -> None:

    # Root Project
    root = Path(__file__).parent.parent.parent

    model_path = root / "models/yolov8m.pt"
    video_file = root / "data/raw/track_video_vehicles.mp4"
    output_dir = root / "data/processed"

    # 2=car, 3=motocycle, 7=truck
    classes_to_count = [2, 3, 7]

    line_points = [(20, 400), (1500, 400)]

    traffic = TrafficCounter(
        model_path,
        video_file,
        output_dir,
        classes_to_count,
        line_points,
    )

    traffic.start_tracking()


if __name__ == "__main__":
    run()
