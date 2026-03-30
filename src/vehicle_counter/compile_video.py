from pathlib import Path

import cv2
from ultralytics import YOLO


class TrafficCounter:
    def __init__(
        self,
        model_path: Path,
        video_source: Path,
        output_dir: Path,
        classes_to_count: list[int],
        line_points: list[tuple[int, int]],
    ) -> None:

        self.model_path = model_path
        self.video_source = video_source
        self.output_dir = output_dir
        self.class_to_count = classes_to_count
        self.line_points = line_points

        if not self.model_path.exists():
            msg = "Modelo não encontrado"
            raise FileNotFoundError(msg)

        if not self.video_source.exists():
            msg = "Video para compilação não encontrado"
            raise FileNotFoundError(msg)

        self.output_dir.mkdir(parents=True, exist_ok=True)

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

    def run(self, file_name: str = "object_output_result") -> None:

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
                    source=frame, persist=True, classes=self.class_to_count, conf=0.5
                )

                for result in results:
                    boxes = result.boxes.cpu().numpy()  # type: ignore

                    if boxes.id is None:
                        continue

                    print("VEHICLE:", boxes.id)
                    for track_id, xyxy, cls_id, conf in zip(
                        boxes.id, boxes.xyxy, boxes.cls, boxes.conf, strict=False
                    ):
                        track_id = int(track_id)
                        cls_id = int(cls_id)
                        conf = float(conf)

                        _x1, _y1, _x2, _y2 = xyxy

                        # Calculate bottom-center coordinate for intersection accuracy
                        x, y = ((_x1 + _x2) / 2), _y2

                        # Retrieve T - 1 spatial state to establish movement vector
                        previous_point = self.track_history.get(track_id)

                        self.track_history[track_id] = (x, y)

                        if previous_point is not None:
                            _x_old, _y_old = previous_point

                        _current_time = frame_counter / fps

                for key, value in self.track_history.items():
                    print(f"ID: {key} | VALUE: {value}")

                frame_counter += 1

        finally:
            ...


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
        model_path, video_file, output_dir, classes_to_count, line_points
    )

    file_name_result = "video_result"
    traffic.run(file_name_result)


if __name__ == "__main__":
    run()
