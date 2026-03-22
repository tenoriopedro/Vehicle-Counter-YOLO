import time
from pathlib import Pathsit

import cv2
from ultralytics import YOLO
from ultralytics.solutions import ObjectCounter


class TrafficCounter:

    def __init__(
            self,
            model_path: Path,
            video_source: Path,
            output_dir: Path) -> None:

        self.model_path = model_path
        self.video_source = video_source
        self.output_dir = output_dir

        if not self.model_path.exists():
            msg = "Modelo não encontrado"
            raise FileNotFoundError(msg)

        if not self.video_source.exists():
            msg = "Video para compilação não encontrado"
            raise FileNotFoundError(msg)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Model
        self.model = YOLO(self.model_path)

        # Classes to be detected (COCO IDs)
        # 2= Car ; 3= Motorcycle ; 5= Bus ; 7= Truck
        self.class_to_count = [2, 3, 5, 7]

        # Line coordinates for tracking
        self.line_points = [(20, 400), (1500, 400)]

        """# Initialize Object Counter
        self.counter = ObjectCounter(
            region=self.line_points,
            model=self.model,
            show_in=True,
            show_out=True,
            classes=self.class_to_count

        )"""


    def run(self, file_name: str = "object_output_result") -> None:

        cap = cv2.VideoCapture(str(self.video_source))

        width, height, fps = (
            int(cap.get(x))
            for x in (
                cv2.CAP_PROP_FRAME_WIDTH,
                cv2.CAP_PROP_FRAME_HEIGHT,
                cv2.CAP_PROP_FPS
            )
        )

        output_result = (self.output_dir / file_name).with_suffix(".mp4")

        video_writer = cv2.VideoWriter(
            str(output_result),
            cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore
            fps,
            (width, height)
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
                    frame,
                    persist=True,
                    classes=self.class_to_count
                )


        finally:
            cap.release()
            video_writer.release()
            cv2.destroyAllWindows()


def run() -> None:
    start = time.time()

    # Root Project
    root = Path(__file__).parent.parent.parent

    model_path = root / "models/yolov8m.pt"
    video_file = root / "data/raw/track_video_vehicles.mp4"
    output_dir = root / "data/processed"

    traffic = TrafficCounter(model_path, video_file, output_dir)

    file_name_result = "video_result"
    traffic.run(file_name_result)

    end = time.time()
    total_time = end - start

    print(f"Inference Time: {total_time:.1f} seconds")


if __name__ == "__main__":
    run()
