from pathlib import Path

from vehicle_counter.analytics.batch_processors import aggregate_unique_tracks
from vehicle_counter.analytics.validators import sanitize_counts
from vehicle_counter.domain.entities import Direction, TrafficReportDTO, VehicleClass


def generate_report(data_dir: Path) -> TrafficReportDTO:
    """
    Orchestrates the extraction and validation of traffic data from Parquet files.

    Args:
        data_dir: Directory containing the processed .parquet files.

    Returns:
        TrafficReportDTO: A strongly typed data transfer object
        containing the final counts and anomalies.

    Raises:
        FileNotFoundError: If the directory does not exist or contains no Parquet files.
    """

    if not data_dir.exists():
        msg = f"Error: Directory {data_dir!r} does not exist."
        raise FileNotFoundError(msg)

    parquet_files = list(data_dir.rglob("*.parquet"))

    if not parquet_files:
        msg = f"Error: No .parquet data found in {data_dir!r}"
        raise FileNotFoundError(msg)

    vehicle_counts, direction_counts = aggregate_unique_tracks(
        parquet_files=parquet_files,
        vehicle_column="class_id",
        direction_column="direction",
    )

    report_vehicles, anomalies_vehicles = sanitize_counts(vehicle_counts, VehicleClass)
    report_directions, anomalies_directions = sanitize_counts(
        direction_counts, Direction
    )

    total_anomalies = anomalies_vehicles + anomalies_directions

    return TrafficReportDTO(report_vehicles, report_directions, total_anomalies)
