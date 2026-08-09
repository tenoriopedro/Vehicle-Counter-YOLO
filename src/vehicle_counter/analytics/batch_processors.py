from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq


def aggregate_unique_tracks(
    parquet_files: Iterable[Path],
    vehicle_column: str,
    direction_column: str,
    track_column: str = "track_id",
) -> tuple[dict[int, int], dict[int, int]]:
    """
    Iterates over Parquet files in chunks, extracts unique
    track IDs per category,
    and returns the aggregated counts in a single pass to minimize I/O overhead.

    Args:
        parquet_files: An iterable of file paths pointing to Parquet data.
        vehicle_column: Column name representing the vehicle classification ID.
        direction_column: Column name representing the direction ID.
        track_column: Column name representing the unique vehicle track ID.

    Returns:
        A tuple containing two dictionaries:
        - vehicle counts (mapping raw ID to total count)
        - direction counts (mapping raw ID to total count)
    """
    global_vehicle_tracks: dict[int, set[int]] = defaultdict(set)
    global_direction_tracks: dict[int, set[int]] = defaultdict(set)

    for file_path in parquet_files:
        parquet_file = pq.ParquetFile(file_path)

        for raw_batch in parquet_file.iter_batches(  # type: ignore
            batch_size=65536, columns=[vehicle_column, direction_column, track_column]
        ):
            # Enforce the C++ to Python boundary for static analysis
            batch = cast("pa.RecordBatch", raw_batch)

            vehicles = cast("list[int]", batch[vehicle_column].to_pylist())
            directions = cast("list[int]", batch[direction_column].to_pylist())
            tracks = cast("list[int]", batch[track_column].to_pylist())

            for v_id, d_id, t_id in zip(vehicles, directions, tracks, strict=False):
                global_vehicle_tracks[v_id].add(t_id)
                global_direction_tracks[d_id].add(t_id)

    vehicles_unique_tracks = {
        key: len(value) for key, value in global_vehicle_tracks.items()
    }
    directions_unique_tracks = {
        key: len(value) for key, value in global_direction_tracks.items()
    }
    return vehicles_unique_tracks, directions_unique_tracks
