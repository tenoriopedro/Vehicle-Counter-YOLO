import json

from confluent_kafka import Consumer


class TelemetryConsumer:
    def __init__(self, bootstrap_server: str, topic: str, group_id: str) -> None:

        self.bootstrap_server = bootstrap_server
        self.topic = topic
        self.group_id = group_id

        self.conf = {
            "bootstrap.servers": self.bootstrap_server,
            "group.id": self.group_id,
            "auto.offset.reset": "earliest",
        }

        self.consumer = Consumer(self.conf)

    def start_consuming(self) -> None:
        self.consumer.subscribe([self.topic])

        try:
            while True:
                message = self.consumer.poll(timeout=1.0)

                if message is None:
                    continue

                if message.error():
                    print(f"Erro while consuming message: {message.error()}")
                    continue

                raw_value = message.value()
                if raw_value is None:
                    continue

                raw_string = raw_value.decode("utf-8")
                data = json.loads(raw_string)

                print(data)

        except KeyboardInterrupt:
            pass

        finally:
            self.consumer.close()


if __name__ == "__main__":
    print("A iniciar consumidor de telemetria..")

    consumer = TelemetryConsumer(
        bootstrap_server="localhost:9092",
        topic="vehicle-telemetry",
        group_id="s3-ingestion-group",
    )
    consumer.start_consuming()
