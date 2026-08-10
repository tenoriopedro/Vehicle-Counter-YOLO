import argparse
import sys
from pathlib import Path

from vehicle_counter.io.config import save_config
from vehicle_counter.io.video import load_frame
from vehicle_counter.presentation.draw_lines import LineCalibrator


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
        "-v",
        "--video",
        type=Path,
        help="Path to the video file to calibrate",
        required=True,
    )

    args = parser.parse_args()

    if not args.video.exists():
        print(f"Error: Video file '{args.video}' not found.", file=sys.stderr)
        return 1

    json_file = args.video.with_suffix(".json")

    frame = load_frame(args.video)

    calibrator = LineCalibrator(window_name=f"Calibration: {args.video.name}")
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
