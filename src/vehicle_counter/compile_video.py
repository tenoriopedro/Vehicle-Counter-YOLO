import time
from pathlib import Path

import cv2
from ultralytics import YOLO
from ultralytics.solutions import ObjectCounter


class TrafficCounter:

    def __init__(self,root_project: Path, model_path: Path, video_source: Path) -> None:
        self.root_project = root_project
        self.model_path = model_path
        self.video_source = video_source

        if not self.model_path.exists():
            msg = "Modelo não encontrado"
            raise FileNotFoundError(msg)

        if not self.video_source.exists():
            msg = "Video para compilação não encontrado"
            raise FileNotFoundError(msg)

        self.result_folder = self.root_project / "data/processed"
        self.result_folder.mkdir(parents=True, exist_ok=True)

        self.model = YOLO(self.root_project / "models/yolov8m.pt")

        # Classes to be detected (COCO IDs)
        self.class_to_count = [2, 3, 7]

        # Line coordinates for tracking
        self.line_points = [(20, 400), (1500, 400)]

        # # Initialize Object Counter
        self.counter = ObjectCounter(
            region=self.line_points,
            model=self.model,
            show_in=True,
            show_out=True,
            classes=self.class_to_count

        )


    def run(self, file_name: str) -> None:

        cap = cv2.VideoCapture(str(self.video_source))

        width, height, fps = (
            int(cap.get(x))
            for x in (
                cv2.CAP_PROP_FRAME_WIDTH,
                cv2.CAP_PROP_FRAME_HEIGHT,
                cv2.CAP_PROP_FPS
            )
        )

        output_result = (self.result_folder / file_name).with_suffix(".mp4")

        video_writer = cv2.VideoWriter(
            str(output_result),
            cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore
            fps,
            (width, height)
        )

        cv2.namedWindow("Video Test", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Video Test", 1200, 800)

        try:

            while True:
                success, frame = cap.read()
                if not success:
                    print(
                        "Video frame is empty "
                        "or video processing has been successfully completed."
                    )
                    break

                results = self.counter(frame)

                results = results.plot_im  # type: ignore

                video_writer.write(results)
                cv2.imshow("Video Test", results)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break


        finally:
            cap.release()
            video_writer.release()
            cv2.destroyAllWindows()
            print(self.counter.counted_ids)


def run() -> None:
    start = time.time()

    # Root Projeto
    root = Path(__file__).parent.parent.parent

    video_file = root / "data/raw/track_video_vehicles.mp4"

    model_path = root / "models/yolov8x.pt"

    file_name = "object_couting_output"

    traffic = TrafficCounter(root, model_path, video_file)
    traffic.run(file_name)

    fim = time.time()
    tempo_total = fim - start

    print(f"Tempo total: {tempo_total:.1f} segundos")


if __name__ == "__main__":
    run()
