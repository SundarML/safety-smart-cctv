from __future__ import annotations

import re
import subprocess
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

    For RTSP sources, uses an FFmpeg subprocess (via imageio-ffmpeg) that
    outputs raw yuv420p which is then converted to BGR via cv2.cvtColor.
    This avoids color corruption seen with HikVision DVR H.264 streams when
    using FFmpeg's internal bgr24 conversion.
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
        self._ffmpeg_proc: subprocess.Popen | None = None
        self._frame_w: int = 0
        self._frame_h: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> "VideoStream":
        if self._is_rtsp():
            self._start_ffmpeg()
        else:
            backend = cv2.CAP_FFMPEG if isinstance(self.source, str) else cv2.CAP_ANY
            self._cap = cv2.VideoCapture(self.source, backend)
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
        if self._ffmpeg_proc:
            self._ffmpeg_proc.terminate()
            try:
                self._ffmpeg_proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self._ffmpeg_proc.kill()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _is_rtsp(self) -> bool:
        return isinstance(self.source, str) and self.source.lower().startswith("rtsp://")

    def _start_ffmpeg(self) -> None:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        # Probe stream — anchor regex to "Video:" so we match the stream
        # resolution, not an earlier pattern in FFmpeg's version/codec output.
        probe = subprocess.run(
            [ffmpeg_exe, "-rtsp_transport", "tcp", "-i", self.source],
            capture_output=True, text=True, timeout=15,
        )
        match = re.search(r"Video:.*?(\d{3,4})x(\d{3,4})", probe.stderr)
        if not match:
            raise RuntimeError(
                f"Cannot determine stream dimensions for: {self.source!r}\n"
                f"FFmpeg output:\n{probe.stderr[-1000:]}"
            )
        self._frame_w = int(match.group(1))
        self._frame_h = int(match.group(2))

        # Output yuv420p and convert to BGR in Python — avoids the color
        # corruption (all-green frames) that occurs when HikVision H.264
        # streams are piped directly as bgr24 through FFmpeg's converter.
        self._ffmpeg_proc = subprocess.Popen(
            [
                ffmpeg_exe,
                "-rtsp_transport", "tcp",
                "-fflags", "nobuffer",
                "-flags", "low_delay",
                "-err_detect", "ignore_err",
                "-i", self.source,
                "-f", "rawvideo",
                "-pix_fmt", "yuv420p",
                "-s", f"{self._frame_w}x{self._frame_h}",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def _capture_loop(self) -> None:
        if self._ffmpeg_proc is not None:
            self._ffmpeg_capture_loop()
        else:
            self._cv2_capture_loop()

    def _ffmpeg_capture_loop(self) -> None:
        # yuv420p is 1.5 bytes per pixel; reshape to (H*3//2, W) for cvtColor.
        yuv_size = self._frame_w * self._frame_h * 3 // 2
        assert self._ffmpeg_proc is not None
        while not self._stop_event.is_set():
            raw = self._ffmpeg_proc.stdout.read(yuv_size)
            if len(raw) < yuv_size:
                self._stop_event.set()
                break
            yuv = np.frombuffer(raw, dtype=np.uint8).reshape(
                self._frame_h * 3 // 2, self._frame_w
            )
            # When a camera is in IR/night mode the DVR sets chroma to 0
            # instead of neutral 128, which makes cvtColor produce a green
            # image. Detect this and return greyscale BGR instead.
            u_mean = float(yuv[self._frame_h:self._frame_h + self._frame_h // 4].mean())
            if u_mean < 10:
                frame = cv2.cvtColor(yuv[:self._frame_h], cv2.COLOR_GRAY2BGR)
            else:
                frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except Empty:
                    pass
            self._queue.put(frame)

    def _cv2_capture_loop(self) -> None:
        consecutive_failures = 0
        max_failures = 10
        while not self._stop_event.is_set():
            assert self._cap is not None
            ret, frame = self._cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    self._stop_event.set()
                    break
                continue
            consecutive_failures = 0
            # Drop the oldest frame when the queue is full so we stay live.
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except Empty:
                    pass
            self._queue.put(frame)
