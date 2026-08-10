import threading
from queue import Empty, Full, Queue

import cv2
import numpy as np
from cv2.typing import MatLike


class VideoStreamer:
    """
    Thread-safe video producer.
    Decouples disk I/O from the inference engine.
    """

    def __init__(self, video_path: str, queue_size: int = 60) -> None:
        self.video_path = video_path
        self.stream = cv2.VideoCapture(video_path)

        if not self.stream.isOpened():
            msg = f"Failed to open video: {video_path}"
            raise FileNotFoundError(msg)

        # Backpressure: Limits RAM consumption by capping the queue size
        self.queue: Queue[MatLike] = Queue(maxsize=queue_size)
        self.stopped: bool = False
        self.thread: threading.Thread | None = None

    def start(self) -> "VideoStreamer":
        """Starts the dedicated disk reading thread."""
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _update(self) -> None:
        """Infinite reading loop. Terminates when the video ends or is forced."""
        while not self.stopped:
            try:
                if not self.queue.full():
                    grabbed, frame = self.stream.read()

                    if not grabbed:
                        self.stop()
                        return

                    self.queue.put(frame, timeout=2.0)
            except Full:
                continue
            except Exception as error:  # noqa: BLE001
                # Logging the error prevents blind exception catching (BLE001)
                print(f"CRITICAL: Error in video stream thread: {error}")
                self.stop()
                return

    def read(self) -> MatLike | None:
        """
        Called by the consumer.
        Returns the frame or None if the video has ended.
        """
        try:
            return self.queue.get(timeout=1.0)
        except Empty:
            if self.stopped:
                return None
            # Safe dummy frame in case of minor disk latency
            return np.zeros((1, 1, 3), dtype=np.uint8)

    def stop(self) -> None:
        """Graceful shutdown of the video stream."""
        self.stopped = True
        if self.stream.isOpened():
            self.stream.release()
