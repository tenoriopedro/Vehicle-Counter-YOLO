import argparse
import shutil
import sys
import time
import uuid
from pathlib import Path

from vehicle_counter.io.config import load_config, save_config
from vehicle_counter.io.sink import TelemetrySink
from vehicle_counter.io.video import get_video_fps, load_frame
from vehicle_counter.presentation.draw_lines import LineCalibrator
from vehicle_counter.services.traffic_analyzer import TrafficCounter
from vehicle_counter.vision.adapters import build_yolo_stream


def run(
    model_path: Path,
    video_path: Path,
    output_dir: Path,
    classes_to_count: list[int],
) -> None:
    """
    Orchestrates the calibration, inference, and telemetry storage pipeline.
    """

    json_file = video_path.with_suffix(".json")
    fps = get_video_fps(video_path)

    # Calibration Phase
    line_points = load_config(json_file)
    if line_points is None:
        video_frame = load_frame(video_path)
        line_points = LineCalibrator().run(video_frame)

        if len(line_points) != 2:
            sys.exit("Calibration aborted by user. Exiting")

        save_config(json_file, line_points)

    # Execution Setup (Staging Area)
    run_id = uuid.uuid4().hex[:8]
    staging_dir = output_dir / f".tmp_{run_id}"
    staging_dir.mkdir(parents=True, exist_ok=True)

    final_dir = output_dir / f"run_{run_id}"
    start_time = time.time()

    # Processing Phase (Context Manager handles safe data flushing)
    try:
        frames_of_objects = build_yolo_stream(model_path, video_path, classes_to_count)

        # The sink now correctly points to the staging directory
        with TelemetrySink(staging_dir) as sink:
            traffic = TrafficCounter(frames_of_objects, fps, sink, line_points)
            traffic.start_tracking()

        # Atomic commit: Rename only if the processing block finishes without errors
        staging_dir.rename(final_dir)

        duration = time.time() - start_time

        print("-" * 30)
        print("PROCESSAMENTO FINALIZADO")
        print(f"Execution ID: run_{run_id}")
        print(f"Total Time: {duration:.2f} seconds")
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
        "-m",
        "--model",
        type=Path,
        help="Path to the YOLO model weights file (e.g., yolov8n.pt).",
        required=True,
    )
    parser.add_argument(
        "-v",
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
        print("Error: Model weights file not found", file=sys.stderr)
        return 1

    if not args.video.exists():
        print("Error: Input video file not found", file=sys.stderr)
        return 1

    # Let unexpected exceptions crash the program so the traceback is visible.
    # We only catch known domain errors gracefully.
    try:
        run(args.model, args.video, args.output, args.cls)

    except KeyboardInterrupt:
        print("\nProcess interrupted by user", file=sys.stderr)
        return 1

    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
