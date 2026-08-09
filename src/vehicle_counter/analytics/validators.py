import math
from enum import IntEnum
from typing import Any, TypeVar

T = TypeVar("T", bound=IntEnum)


def sanitize_counts(
    counts: dict[Any, int], entity_class: type[T]
) -> tuple[dict[T, int], int]:
    """
    Acts as an anti-corruption layer, filtering out invalid or corrupt keys
    before they reach the domain level.

    Args:
        counts: A dictionary containing raw extracted keys and their occurrences.
        entity_class: The IntEnum domain class to validate the keys against.

    Returns:
        A tuple containing:
        - A dictionary strictly typed with the domain IntEnum instances as keys.
        - An integer representing the total sum of anomalies (failed records).
    """

    clean_counts: dict[T, int] = {}
    anomalies = 0

    for data_raw, count in counts.items():
        if not isinstance(data_raw, (int, float)) or (
            isinstance(data_raw, float) and math.isnan(data_raw)
        ):
            anomalies += count
            continue
        try:
            clean_data = entity_class(int(data_raw))
            clean_counts[clean_data] = clean_counts.get(clean_data, 0) + count
        except ValueError:
            anomalies += count

    return clean_counts, anomalies
