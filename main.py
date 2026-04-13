import argparse
import sys
from pathlib import Path

from vehicle_counter.tracker import TrafficCounter


def run(
    model: Path,
    video: Path,
    output: Path,
    cls: list[int],
) -> None:

    model_path = model
    video_file = video
    output_dir = output
    classes_to_count = cls

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
    parser = argparse.ArgumentParser(
        description="""
        Pipeline for bidirectional
        vehicle extraction and counting using YOLO models.
        """
    )

    parser.add_argument(
        "--model",
        type=Path,
        help="Path to the YOLO model weights file (e.g., yolov8n.pt).",
        required=True,
    )
    parser.add_argument(
        "--video",
        type=Path,
        help="Path to the input video file to be processed.",
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="""
        Directory path to save the processed Parquet files.
        (default: data/processed)
        """,
        default=Path("data/processed"),
    )
    parser.add_argument(
        "--cls",
        nargs="+",
        type=int,
        help="""
        List of COCO dataset class IDs to track.
        Default is 2 3 7 (2=car, 3=motorcycle, 7=truck).
        """,
        default=[2, 3, 7],
    )

    args = parser.parse_args()

    if not args.model.exists():
        msg = "Model Not Found"
        sys.exit(msg)

    if not args.video.exists():
        msg = "Video Not Found"
        sys.exit(msg)

    result = run(args.model, args.video, args.output, args.cls)
