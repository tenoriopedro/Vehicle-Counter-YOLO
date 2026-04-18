import argparse
import shutil
import sys
import uuid
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

    run_id = uuid.uuid4().hex[:8]

    staging_dir = output_dir / f".tmp_{run_id}"

    traffic = TrafficCounter(
        model_path,
        video_file,
        staging_dir,
        classes_to_count,
        line_points,
    )
    try:
        traffic.start_tracking()

        staging_dir.rename(output_dir / f"run_{run_id}")

    except Exception as e:  # noqa: BLE001
        shutil.rmtree(staging_dir, ignore_errors=True)
        sys.exit(f"Erro critico no processamento: {e}")


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
