import argparse
import shutil
import sys
import time
import uuid
from pathlib import Path

from pydantic import BaseModel, Field, StrictInt, ValidationError

from vehicle_counter.tracker import TrafficCounter


def load_and_validate_config(json_file: Path) -> list[tuple[int, int]]:
    """
    Loads and validates the intersection line JSON configuration file.

    Uses Pydantic to ensure the file exists, has valid JSON syntax, and strictly
    adheres to the geometric contract (an array of exactly two points, composed
    of integers).

    Args:
        json_file (Path): The path to the JSON calibration file.

    Returns:
        list[tuple[int, int]]: A list containing exactly two tuples, representing
        the (X, Y) coordinates of Point A and Point B.

    Raises:
        SystemExit: Aborts the process (exit code 1) and outputs an error to stderr
        if the file is missing, invalid, or violates the required geometry.
    """

    class CalibrationConfig(BaseModel):
        line_points: list[tuple[StrictInt, StrictInt]] = Field(
            min_length=2, max_length=2
        )

    if not json_file.exists():
        print(
            f"Error: Missing calibration file '{json_file.name}'.\n"
            "The intersection line coordinates are required before counting.\n"
            f"Fix this by running: vehicle-calibrate --video "
            f"{json_file.with_suffix('.mp4')}",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        config = CalibrationConfig.model_validate_json(json_file.read_text())

    except ValidationError as e:
        print(
            f"Error: Invalid configuration in {json_file.name}.\nDetails: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    return config.line_points


def run(
    model: Path,
    video: Path,
    output: Path,
    cls: list[int],
) -> None:
    """
    Orchestrates the setup, execution, and teardown of the tracking process.

    Args:
        model (Path): File path to the YOLO model.
        video (Path): File path to the target video.
        output (Path): Base directory for exported Parquet files.
        cls (list[int]): List of YOLO class IDs to detect.
    """

    model_path = model
    video_file = video
    output_dir = output
    classes_to_count = cls

    json_file = video_file.with_suffix(".json")

    line_points = load_and_validate_config(json_file)

    run_id = uuid.uuid4().hex[:8]

    # Staging directory prevents corrupt outputs if the script crashes midway
    staging_dir = output_dir / f".tmp_{run_id}"

    start_time = time.time()

    traffic = TrafficCounter(
        model_path,
        video_file,
        staging_dir,
        classes_to_count,
        line_points,
        show_video_window=True
    )
    try:
        traffic.start_tracking()

        staging_dir.rename(output_dir / f"run_{run_id}")

        end_time = time.time()
        duration = end_time - start_time

        print("-" * 30)
        print("PROCESSAMENTO FINALIZADO")
        print(f"ID da Execução: run_{run_id}")
        print(f"Tempo Total: {duration:.2f} segundos")
        print("-" * 30)

    except Exception as e:  # noqa: BLE001
        shutil.rmtree(staging_dir, ignore_errors=True)
        sys.exit(f"Erro critico no processamento: {e}")


def main() -> int:
    """
    Parses command-line arguments and triggers the execution pipeline.

    Returns:
        int: System exit code (0 for success, 1 for failure).
    """

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
        help="""
        Path to the input video file to be processed.
        Pre-calibration of the video is required
        (execute 'vehicle-calibrate --video <path>'
        to create the necessary .json file)
        """,
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
        print("Error: Model Not Found", file=sys.stderr)
        return 1

    if not args.video.exists():
        print("Error: Video Not Found", file=sys.stderr)
        return 1

    try:
        run(args.model, args.video, args.output, args.cls)

    except Exception as e:  # noqa: BLE001
        print(f"Critical execution error: {e}", file=sys.stderr)
        return 1

    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
