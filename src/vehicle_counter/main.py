import argparse
import shutil
import sys
import time
import uuid
from pathlib import Path

from vehicle_counter.io.config import load_config
from vehicle_counter.io.sink import TelemetrySink
from vehicle_counter.io.video import get_video_fps
from vehicle_counter.services.traffic_analyzer import TrafficCounter
from vehicle_counter.vision.adapters import build_yolo_stream

DATA_PATH = Path.cwd() / "data/raw"


def run(
    model_path: Path,
    context_name: Path,
    output_dir: Path,
    classes_to_count: list[int],
) -> None:
    """
    Orchestrates the inference and telemetry storage pipeline.
    Expects a strict workspace structure: data/raw/<context>/<context>.[mp4|json]
    """

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

    # Dependency Loading
    fps = get_video_fps(video_path)
    line_points = load_config(json_file)

    if line_points is None or len(line_points) != 2:
        msg = (
            "Critical Error: "
            f"Calibration file at {json_file} is corrupted or incomplete."
        )
        raise ValueError(msg)

    # Execution Setup (Staging Area)
    run_id = uuid.uuid4().hex[:8]
    staging_dir = output_dir / f".tmp_{run_id}"
    staging_dir.mkdir(parents=True, exist_ok=True)
    final_dir = output_dir / f"run_{run_id}"

    start_time = time.time()

    # Processing Phase
    try:
        frames_of_objects = build_yolo_stream(model_path, video_path, classes_to_count)

        # The sink now correctly points to the staging directory
        with TelemetrySink(staging_dir) as sink:
            traffic = TrafficCounter(frames_of_objects, fps, sink, line_points)
            traffic.start_tracking()

        staging_dir.rename(final_dir)

        duration = time.time() - start_time

        print("-" * 30)
        print("PROCESSING COMPLETED")
        print(f"Execution ID: run_{run_id}")
        print(
            "Execution Time: "
            f"{time.strftime('%H hours %M minutes %S seconds', time.gmtime(duration))}"
        )
        print("-" * 30)

    except (KeyboardInterrupt, Exception):
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def main() -> int:
    """
    Parses command-line arguments and triggers the execution pipeline.
    """

    parser = argparse.ArgumentParser(
        description="""
        Pipeline for bidirectional
        vehicle extraction and counting using YOLO models.
        """
    )

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
        print("Error: Model weights file not found", file=sys.stderr)
        return 1

    try:
        run(args.model, args.context, args.output, args.cls)

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
