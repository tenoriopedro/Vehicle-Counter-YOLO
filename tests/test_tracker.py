from unittest.mock import MagicMock, patch

import pytest

from vehicle_counter.tracker import TrafficCounter


@patch("vehicle_counter.tracker.YOLO")
@pytest.mark.parametrize(
    ("previous_point", "y_bottom", "expected"),
    (
        ((500, 380), 410, 0),
        ((100, 420), 390, 1),
        ((290, 350), 380, None),
    ),
)
def test_traffic_counter(
    mock_yolo: MagicMock,
    previous_point: tuple[int, int],
    y_bottom: int,
    expected: int | None) -> None:

    model = MagicMock()
    video_source = MagicMock()
    output_dir = MagicMock()
    class_to_count = [0, 1]
    line_point = [(20, 400), (1500, 400)]

    model.exists.return_value = True
    video_source.exists.return_value = True

    counter = TrafficCounter(
        model,
        video_source,
        output_dir,
        class_to_count,
        line_point
    )

    assert counter._check_intersection_point(previous_point, y_bottom) == expected


@patch("pandas.DataFrame.to_parquet")
@patch("vehicle_counter.tracker.YOLO")
def test_flush_clears_memory(
    mock_yolo: MagicMock,
    mock_pandas: MagicMock) -> None:

    model = MagicMock()
    video_source = MagicMock()
    output_dir = MagicMock()
    class_to_count = [0, 1]
    line_point = [(20, 400), (1500, 400)]

    model.exists.return_value = True
    video_source.exists.return_value = True

    counter = TrafficCounter(
        model,
        video_source,
        output_dir,
        class_to_count,
        line_point
    )

    for key in counter.detected_data:
        counter.detected_data[key] = [1, 2, 3]

    counter._flush_to_disk(0.0)

    assert len(counter.detected_data["timestamp"]) == 0
    assert counter.batch_counter == 2
