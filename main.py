from pathlib import Path

from src.vehicle_counter.tracker import TrafficCounter


def run() -> None:

    # Root Project
    root = Path(__file__).parent

    model_path = root / "models/yolov8m.pt"
    video_file = root / "data/raw/track_video_vehicles.mp4"
    output_dir = root / "data/processed"

    # 2=car, 3=motocycle, 7=truck
    classes_to_count = [2, 3, 7]

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
    run()
