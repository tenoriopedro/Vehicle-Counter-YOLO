from vehicle_counter.domain.entities import TrafficReportDTO


def print_traffic_report(report: TrafficReportDTO) -> None:

    print("=== TRAFFIC REPORT ===")
    total_vehicles = 0
    for entity, count in report.valid_classes_counts.items():
        print(f"{entity.label}: {count}")
        total_vehicles += count

    print("-" * 30)
    print(f"Total Number of Vehicles: {total_vehicles}")

    print("\n=== DIRECTIONAL FLOW ===")
    for direction, count in report.valid_directions_counts.items():
        print(f"{direction.label}: {count}")

    print("\n=== ANOMALIES ===")
    print(f"Total Tracking Errors: {report.total_anomalies}\n")
