from pathlib import Path

import cv2
from cv2.typing import MatLike


def load_frame(video_path: Path) -> MatLike:
    """
    Safely opens a video stream, extracts the initial frame for calibration purposes,
    and immediately releases the hardware resources.

    Args:
        video_path: Path to the target video file.

    Returns:
        MatLike: The first frame of the video as an OpenCV matrix.

    Raises:
        FileNotFoundError: If the video path does not exist.
        ValueError: If OpenCV fails to decode the frame.
    """

    if not video_path.exists():
        msg = f"Error: The file '{video_path}' was not found."
        raise FileNotFoundError(msg)

    cap = cv2.VideoCapture(str(video_path))
    success, frame = cap.read()
    cap.release()

    if not success:
        msg = "Error: Could not read the first frame of the video."
        raise ValueError(msg)

    return frame


def get_video_fps(video_path: Path) -> float:
    """Extracts the exact frames-per-second metadata from the video."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps <= 0:
        msg = f"Failed to extract video metadata. Got FPS={fps}."
        raise ValueError(msg)

    return float(fps)
