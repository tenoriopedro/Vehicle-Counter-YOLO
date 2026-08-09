import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from vehicle_counter.domain.events import VehicleEvent


class TelemetrySink:
    """
    Buffers vehicle tracking events in memory and periodically flushes them
    to disk as Parquet files to optimize I/O operations.
    Must be used as a context manager to prevent data loss on exit.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.flush_limit: int = 50000
        self.event_list: list[VehicleEvent] = []

        self.schema = pa.schema(
            [
                ("track_id", pa.int32()),
                ("class_id", pa.int8()),
                ("confidence", pa.float32()),
                ("direction", pa.int8()),
                ("timestamp", pa.timestamp("us", tz="UTC")),
            ]
        )

    def __enter__(self) -> "TelemetrySink":
        """Initializes the context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Ensures any remaining events in the buffer
        are flushed upon exiting the context."""
        self.flush()

    def add(self, event: VehicleEvent) -> None:
        """
        Appends a new event to the internal buffer.
        Triggers a disk write if the buffer capacity is reached.
        """

        self.event_list.append(event)

        if len(self.event_list) >= self.flush_limit:
            self.flush()

    def flush(self) -> None:
        """
        Serializes the buffered events into a PyArrow Table and writes them
        to a timestamped Parquet file, then clears the memory buffer.
        """

        if len(self.event_list) == 0:
            return

        data_dict: list[dict[str, Any]] = [
            event.model_dump() for event in self.event_list
        ]

        pa_table: pa.Table = pa.Table.from_pylist(data_dict, schema=self.schema)

        file_name: str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        file_id: str = uuid.uuid4().hex[:8]
        parquet_file = f"{file_name}_{file_id}.parquet"

        save_parquet_file: Path = self.data_dir / parquet_file

        pq.write_table(pa_table, str(save_parquet_file))  # type: ignore

        self.event_list.clear()
