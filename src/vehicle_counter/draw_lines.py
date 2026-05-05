import argparse
import json
import sys
from pathlib import Path

import cv2


class LineCalibrator:
    """
    Interactive GUI tool to establish and save a counting line
    for vehicle tracking. Encapsulates state and hardware management.
    """

    def __init__(self, video_path: Path) -> None:

        self.video_path = video_path
        self.line_points: list[tuple[int, int]] = []
        self.frame = None
        self.window_name = "Line Calibration Tool"

    def _load_frame(self) -> None:
        """
        Safely opens the video, extracts the first frame, and immediately
        releases the hardware. Raises exceptions on failure.
        """

        if not self.video_path.exists():
            msg = (f"Error: The file '{self.video_path}' was not found.",
            )
            raise FileNotFoundError(msg)

        cap = cv2.VideoCapture(str(self.video_path))
        success, self.frame = cap.read()
        cap.release()

        if not success:
            msg = (
                "Error: Could not read the first frame of the video."
            )
            raise ValueError(msg)

    def handle_click(
            self,
            event: int,
            x: int,
            y: int,
            flags: int,
            param: object) -> None:
        """
        OpenCV mouse callback event handler.
        Note: 'flags' and 'param' are unused but required by
        the cv2.setMouseCallback signature.
        """

        if self.frame is None:
            return

        if event == cv2.EVENT_LBUTTONDOWN:

            print(f"[NEW COORDINATE] X: {x} | Y: {y}")

            if len(self.line_points) < 2:
                self.line_points.append((x, y))

            else:
                self.line_points.clear()
                self.line_points.append((x, y))

            frame_temp = self.frame.copy()

            if len(self.line_points) == 2:
                point1, point2 = self.line_points
                cv2.line(frame_temp, point1, point2, (0,255,0), 5)

            for point in self.line_points:

                # Draw a green dot ai the clicked location for visual feedback
                cv2.circle(frame_temp, point, 5, (0, 255, 0), -1)

            cv2.imshow(self.window_name, frame_temp)

    def save(self) -> None:
        """
        Persists the calibration data to a JSON file if the geometry is valid.
        """
        if len(self.line_points) == 2:
            data = {"line_points": self.line_points}
            json_path = self.video_path.with_suffix(".json")

            with open(json_path, "w") as file:
                json.dump(data, file, indent=4)

            print(f"\nCalibration saved to: {json_path}")

        else:
            print(
                "\nCalibration aborted. Exactly 2 points are requerid.",
                file=sys.stderr
            )

    def run(self) -> int:
        """
        Executes the main application lifecycle: initialization, UI loop,
        and teardown.

        Returns:
            int: Exit code (0 for success, 1 for failure).
        """
        try:
            self._load_frame()
        except Exception as e:  # noqa: BLE001
            print(f"Error {e}", file=sys.stderr)
            return 1

        if self.frame is None:
            msg = "Frame cannot be None at rendering stage."
            raise ValueError(msg)

        cv2.imshow(self.window_name, self.frame)
        cv2.setMouseCallback(self.window_name, self.handle_click)

        print("=" * 40)
        print("CALIBRATION MODE ACTIVE")
        print("1. Click on the image to establish Point A and Point B.")
        print("2. A third click will reset the line.")
        print("3. Press 'q' or 'ESC' on the video window to save and exit.")
        print("=" * 40)

        # UI Render Loop
        while True:
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()
        self.save()
        return 0



def main() -> int:
    """
    Parses command-line arguments and triggers the calibration tool.

    Returns:
        int: System exit code.
    """
    parser = argparse.ArgumentParser(description="Traffic Line Calibration Tool")
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to the input video file"
    )
    args = parser.parse_args()

    calibrator = LineCalibrator(args.video)

    return calibrator.run()


if __name__ == "__main__":
    sys.exit(main())
