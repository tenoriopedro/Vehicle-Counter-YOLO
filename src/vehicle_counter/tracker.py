from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO
from ultralytics.engine.results import Boxes, Results


class TrafficCounter:
    def __init__(
        self,
        model_path: Path,
        video_source: Path,
        output_dir: Path,
        classes_to_count: list[int],
        line_points: list[tuple[int, int]],
        conf: float = 0.1,
        file_name: str = "video_result",
        *,
        show_video_window: bool = False,
    ) -> None:
        """
        Initializes the TrafficCounter instance.

        Args:
            model_path (Path): Path to the compiled YOLO model weights (.pt).
            video_source (Path): Path to the input video file.
            output_dir (Path): Directory where the output Parquet files will be saved.
            classes_to_count (list[int]): COCO class IDs to track (e.g., [2, 3, 7]).
            line_points (list[tuple[int, int]]): Two coordinates defining the
            virtual counting line.
            conf (float, optional): Minimum confidence threshold
            for YOLO detections. Defaults to 0.1.
            file_name (str, optional): Base name for the output Parquet
            files. Defaults to "video_result".
            show_video_window (bool, optional): If True, renders the OpenCV
            video feed with bounding boxes. Useful for debugging but
            should be False for headless production. Defaults to False.

        Raises:
            FileNotFoundError: If the model or video file does not exist.
        """

        self.model_path = model_path
        self.video_source = video_source
        self.output_dir = output_dir
        self.class_to_count = classes_to_count
        self.line_points = line_points
        self.conf = conf
        self.file_name = file_name
        self.show_video_window = show_video_window
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        if not self.model_path.exists():
            msg = "Modelo não encontrado"
            raise FileNotFoundError(msg)

        if not self.video_source.exists():
            msg = "Video para compilação não encontrado"
            raise FileNotFoundError(msg)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        total_files = len(list(self.output_dir.glob("*.parquet")))
        self.batch_counter = total_files + 1

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

        self.last_seen_timestamp: dict[int, float] = {}
        self.ttl_seconds: float = 2.0

        self.flush_limit = 100

    def start_tracking(self) -> None:
        """
        Initiates the video capture and coordinates
        the frame-by-frame tracking pipeline.

        Raises:
            ValueError: If the video metadata (FPS) cannot be correctly extracted.
        """
        cap = cv2.VideoCapture(str(self.video_source))

        fps = cap.get(cv2.CAP_PROP_FPS)

        cap.release()

        if fps <= 0:
            msg = (
                "Falha ao extrair metadados do video. "
                f"Obtido: FPS={fps}\n"
                "Verifique a integridade da fonte de vídeo"
            )

            raise ValueError(msg)

        results = self.model.track(
            source=str(self.video_source),
            stream=True,
            persist=True,
            classes=self.class_to_count,
            conf=self.conf,
            iou=0.45,
            imgsz=608,
            vid_stride=2,
        )

        effective_fps = fps / 2
        frame_counter = 0
        current_time = 0.0

        try:
            for result in results:
                boxes = result.boxes.cpu().numpy()  # type: ignore

                current_time = round(frame_counter / effective_fps, 2)

                self._register_crossings(boxes, current_time)

                if len(self.detected_data["timestamp"]) >= self.flush_limit:
                    self._flush_to_disk(current_time)

                frame_counter += 1

                if self.show_video_window and self.show_video(results):
                    break

        finally:
            self._flush_to_disk(current_time)

    def show_video(self, results: list[Results]) -> bool:

        annotated_frame = results[0].plot()

        cv2.line(
            annotated_frame, self.line_points[0], self.line_points[1], (0, 0, 255), 2
        )

        cv2.imshow("VIDEO SHOW", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Processamento interrompido pelo utilizador.")
            return True

        return False

    def _register_crossings(self, boxes: Boxes, current_time: float) -> None:
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
            boxes.id, boxes.xyxy, boxes.cls, boxes.conf, strict=False
        ):
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

            if direction is not None and track_id not in self.processed_ids:
                self.processed_ids.add(track_id)

                self.detected_data["timestamp"].append(current_time)
                self.detected_data["track_id"].append(track_id)
                self.detected_data["class_id"].append(cls_id)
                self.detected_data["confidence"].append(conf)
                self.detected_data["direction"].append(direction)

                self.last_seen_timestamp[track_id] = current_time

    def _flush_to_disk(self, current_timestamp: float) -> None:
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

        parquet_file = f"{self.file_name}_{self.run_id}_{counter.zfill(3)}.parquet"

        save_parquet_file = self.output_dir / parquet_file

        df.to_parquet(save_parquet_file, engine="pyarrow")

        self.batch_counter += 1

        for key in self.detected_data:
            self.detected_data[key].clear()

        self._cleanup_memory(current_timestamp)

    def _check_intersection_point(
        self, previous_point: tuple[int, int] | None, y_bottom: int
    ) -> int | None:
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

    def _cleanup_memory(self, current_timestamp: float) -> None:

        keys_to_delete = [
            key
            for key, last_seen in self.last_seen_timestamp.items()
            if (current_timestamp - last_seen) > self.ttl_seconds
        ]

        for key in keys_to_delete:
            self.last_seen_timestamp.pop(key, None)
            self.track_history.pop(key, None)
            self.processed_ids.discard(key)
