import argparse
import sys
from pathlib import Path

from vehicle_counter.io.config import save_config
from vehicle_counter.io.video import load_frame
from vehicle_counter.presentation.draw_lines import LineCalibrator

DATA_PATH = Path.cwd() / "data/raw"


def main() -> int:
    """
    CLI entry point for the video calibration tool.
    Extracts a frame, launches the UI, and saves the configuration.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Interactive tool to calibrate the counting line for a specific video."
        )
    )
    parser.add_argument(
        "-c",
        "--context",
        type=Path,
        help="Name of the context folder inside data/raw/",
        required=True,
    )

    args = parser.parse_args()

    context_path = DATA_PATH / args.context
    video_path = context_path / args.context.with_suffix(".mp4")
    json_file = context_path / args.context.with_suffix(".json")

    # Strict Physical Validation (Fail Fast)
    if not video_path.is_file():
        print(f"Critical Error: Video file not found at {video_path}", file=sys.stderr)
        return 1

    frame = load_frame(video_path)

    calibrator = LineCalibrator(window_name=f"Calibration: {args.context.name}")
    line_points = calibrator.run(frame)

    if len(line_points) != 2:
        print(
            "Calibration aborted or incomplete. Exactly two points are required",
            file=sys.stderr,
        )
        return 1

    save_config(json_file, line_points)
    print(f"Calibration successful! Configuration saved to '{json_file}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
