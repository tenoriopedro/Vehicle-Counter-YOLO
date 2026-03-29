import time
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
            line_points: list[tuple[int, int]]) -> None:

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




    def run(self, file_name: str = "object_output_result") -> None:

        cap = cv2.VideoCapture(str(self.video_source))

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            msg = (
                "Falha ao extrair metadados do video. "
                f"Obtido: FPS={fps}\n"
                "Verifique a integridade da fonte de vídeo"
            )
            raise ValueError(
                msg
            )

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
                    classes=self.class_to_count
                )


        finally:
            ...



if __name__ == "__main__":
    ...
