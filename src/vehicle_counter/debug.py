import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

from vehicle_counter.io.config import load_config

DATA_PATH = Path.cwd() / "data/raw"


def run(
    model_path: Path,
    context_name: Path,
) -> None:

    context_path = DATA_PATH / context_name
    video_path = context_path / context_name.with_suffix(".mp4")
    json_file = context_path / context_name.with_suffix(".json")

    # Strict Physical Validation (Fail Fast)
    if not video_path.is_file():
        msg = f"Critical Error: Video file not found at {video_path}"
        raise FileNotFoundError(msg)

    if not json_file.is_file():
        msg = (
            f"Critical Error: Calibration matrix not found at {json_file}. "
            f"Run 'uv run vehicle-calibrate --context {context_name}' first."
        )
        raise FileNotFoundError(msg)

    line_points = load_config(json_file)

    if line_points is None or len(line_points) != 2:
        msg = (
            "Critical Error: "
            f"Calibration file at {json_file} is corrupted or incomplete."
        )
        raise ValueError(msg)

    model_yolo = YOLO(model_path)

    results = model_yolo.track(
        source=str(video_path),
        stream=True,
        conf=0.5,
        imgsz=640,
    )

    for result in results:
        frame = result.orig_img

        boxes = result.boxes

        if boxes is None:
            continue

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        cv2.line(frame, line_points[0], line_points[1], (0, 0, 255), 2)

        cv2.imshow("Video Test", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break

        if cv2.getWindowProperty("Video Test", cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()

def main() -> int:

    parser = argparse.ArgumentParser(description="Teste Visual")

    parser.add_argument(
        "-c",
        "--context",
        type=Path,
        help="""
        Name of the context folder inside data/raw/.
        The system expects data/raw/<context>/<context>.mp4 and <context>.json to exist.
        """,
        required=True,
    )
    parser.add_argument(
        "-m",
        "--model",
        type=Path,
        help="Path to the YOLO model weights file (e.g., yolov8n.pt).",
        required=True,
    )

    args = parser.parse_args()

    if not args.model.exists():
        print("Error: Model weights file not found", file=sys.stderr)
        return 1

    try:
        run(args.model, args.context)

    except KeyboardInterrupt:
        print("\nProcess interrupted by user", file=sys.stderr)
        return 1
    except (FileNotFoundError, ValueError) as e:
        # Graceful handling for known domain errors to avoid ugly stack traces for users
        print(f"\n{e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
