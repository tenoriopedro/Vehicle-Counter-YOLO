import argparse
import shutil
import sys
import time
import uuid
from pathlib import Path

from vehicle_counter.tracker import TrafficCounter


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

    line_points = [(2, 180), (639, 180)]
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
