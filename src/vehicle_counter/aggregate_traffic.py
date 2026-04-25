import argparse
import sys
from pathlib import Path
from typing import cast

import pandas as pd


def generate_report(data_dir: Path) -> int:
    """
    Reads all Parquet files in the specified directory, aggregates tracking IDs,
    and prints a statistical summary of vehicle classes and movement directions.

    Args:
        data_dir (Path): The directory containing the processed .parquet files.

    Returns:
        int: Exit code (0 for success, 1 for failure).
    """
    if not data_dir.exists():
        print(f"Error: Directory {data_dir} does not exist.", file=sys.stderr)
        return 1

    files = list(data_dir.rglob("*.parquet"))

    if not files:
        print(f"Error: No .parquet data found in {data_dir}", file=sys.stderr)
        return 1

    # Warning: Loading all files into RAM simultaneously
    # Acceptable for small/medium datasets, requires chunking for large-scale operation
    df = pd.concat([pd.read_parquet(f) for f in files])

    print("=== RELATÓRIO DE TRÁFEGO ===")

    # Calculate unique track_ids per class to prevent counting the same vehicle twice
    counts = df.groupby("class_id")["track_id"].nunique()

    mapping = {2: "Carros", 3: "Motos", 5: "Autocarros", 7: "Camiões"}

    total_vehicles = 0

    for class_id_raw, count in counts.items():
        class_id = int(cast("int", class_id_raw))

        # Fallback mechanism: if the model hallucinated a class ID, show the raw number
        label = mapping.get(int(class_id), f"Classe {class_id}")
        print(f"{label}: {count}")
        total_vehicles += count

    print(f"Total de veiculos: {total_vehicles}")

    print("\n=== FLUXO POR DIREÇÃO ===")
    direction_counts = df.groupby("direction")["track_id"].nunique()
    dir_mapping = {0: "Sul(esq)", 1: "Norte(dir)"}

    for dir_id_raw, count in direction_counts.items():
        dir_id = int(cast("int", dir_id_raw))

        # Fallback mechanism for unexpected directions
        dir_label = dir_mapping.get(dir_id, f"Direção Desconhecida ({dir_id})")

        print(f"{dir_label}: {count}")

    return 0


def main() -> int:
    """
    Parses command-line arguments and triggers the report generation.

    Returns:
        int: System exit code.
    """
    parser = argparse.ArgumentParser(
        description="Generates a traffic flow report from processed Parquet files."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed"),
        help="Directory path containing the Parquet files. (default: data/processed)",
    )

    args = parser.parse_args()

    return generate_report(args.data)


if __name__ == "__main__":
    sys.exit(main())
