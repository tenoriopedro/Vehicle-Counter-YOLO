from pathlib import Path

from pydantic import BaseModel, Field, StrictInt, ValidationError


class CalibrationConfig(BaseModel):
    """
    Schema for validating the calibration line coordinates.
    Ensures exactly two points (x, y) are provided.
    """

    line_points: list[tuple[StrictInt, StrictInt]] = Field(min_length=2, max_length=2)


def load_config(json_file: Path) -> list[tuple[int, int]] | None:
    """
    Attempts to load and validate line calibration points from a JSON file.

    Args:
        json_file: Path to the configuration file.

    Returns:
        A list of two (x, y) tuples if successful,
        or None if the file is missing or invalid.
    """
    if not json_file.exists():
        return None

    try:
        config = CalibrationConfig.model_validate_json(json_file.read_text())

    except ValidationError:
        return None

    else:
        return config.line_points


def save_config(json_file: Path, line_points: list[tuple[int, int]]) -> None:
    """
    Serializes and saves the calibration line points to a JSON file.

    Args:
        json_file: Destination path for the configuration.
        line_points: A list of exactly two (x, y) coordinates.
    """
    config = CalibrationConfig(line_points=line_points)
    json_file.write_text(config.model_dump_json(indent=4))
