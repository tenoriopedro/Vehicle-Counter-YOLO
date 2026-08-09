import argparse
import sys
from pathlib import Path

from vehicle_counter.analytics.aggregate_traffic import generate_report
from vehicle_counter.presentation.cli_reports import print_traffic_report


def main() -> int:
    """
    CLI entry point. Parses arguments, triggers report generation,
    and handles system-level errors.

    Returns:
        int: 0 for success, 1 for failure.
    """

    parser = argparse.ArgumentParser(
        description="Generates a traffic flow report from processed Parquet files."
    )
    parser.add_argument(
        "-d",
        "--data",
        type=Path,
        default=Path("data/processed"),
        help="Directory path containing the Parquet files. (default: data/processed)",
    )

    args = parser.parse_args()

    try:
        report_dto = generate_report(args.data)
        print_traffic_report(report_dto)

    except FileNotFoundError as err:
        print(str(err), file=sys.stderr)
        return 1

    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
