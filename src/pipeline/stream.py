from __future__ import annotations

import threading
from queue import Empty, Queue
from typing import Union

import cv2
import numpy as np


class VideoStream:
    """
    Reads frames from a camera or video file in a background thread so the
    main thread is never blocked waiting on I/O.  A small queue decouples
    capture speed from processing speed; when the queue is full the oldest
    frame is dropped so the detector always sees the most recent image.
    """

    def __init__(
        self,
        source: Union[int, str],
        width: int | None = None,
        height: int | None = None,
        queue_size: int = 2,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self._queue: Queue[np.ndarray] = Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._cap: cv2.VideoCapture | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> "VideoStream":
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source!r}")
        if self.width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name=f"stream-{self.source}")
        self._thread.start()
        return self

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Return (True, frame) or (False, None) on timeout / stream end."""
        try:
            return True, self._queue.get(timeout=1.0)
        except Empty:
            return False, None

    def is_running(self) -> bool:
        return not self._stop_event.is_set()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._cap:
            self._cap.release()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            assert self._cap is not None
            ret, frame = self._cap.read()
            if not ret:
                self._stop_event.set()
                break
            # Drop the oldest frame when the queue is full so we stay live.
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except Empty:
                    pass
            self._queue.put(frame)
