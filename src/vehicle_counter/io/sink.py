
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from vehicle_counter.domain.events import VehicleEvent


class TelemetrySink:

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.flush_limit: int = 1000
        self.event_list: list[VehicleEvent] = []

        self.schema = pa.schema(
            [
                ("track_id", pa.int32()),
                ("class_id", pa.int8()),
                ("confidence", pa.float32()),
                ("direction", pa.int8()),
                ("timestamp", pa.timestamp("us", tz="UTC"))
            ]
        )

    def add(self, event: VehicleEvent) -> None:

        self.event_list.append(event)

        if len(self.event_list) >= self.flush_limit:
            self._flush()


    def _flush(self) -> None:

        if len(self.event_list) == 0:
            return

        data_dict: list[dict[str, Any]] = [
            event.model_dump() for event in self.event_list
        ]

        pa_table: pa.Table = pa.Table.from_pylist(data_dict, schema=self.schema)

        print()
        print()
        print()
        print()
        print(pa_table)
        print()
        print()

        file_name: str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

        file_id: str = uuid.uuid4().hex[:8]

        parquet_file = f"{file_name}_{file_id}.parquet"

        save_parquet_file: Path = self.data_dir / parquet_file

        pq.write_table(pa_table, str(save_parquet_file)) # type: ignore

        self.event_list.clear()


if __name__ == "__main__":

    from vehicle_counter.domain.events import Direction

    # Preparar o diretório de teste
    test_dir = Path("test_parquet_output")
    test_dir.mkdir(exist_ok=True)

    # Instanciar a tua classe
    sink = TelemetrySink(test_dir)

    # Forçar o limite para testar mais rápido (opcional)
    sink.flush_limit = 10
    print(f"A iniciar teste... Eventos na lista: {len(sink.event_list)}")

    # Injetar dados falsos simulando o YOLO
    for i in range(12):
        dummy_event = VehicleEvent(
            track_id=i,
            class_id=2,
            confidence=0.85,
            direction=Direction.SOUTH
        )
        sink.add(dummy_event)
        print(f"Adicionado ID {i}. Tamanho da lista: {len(sink.event_list)}")
