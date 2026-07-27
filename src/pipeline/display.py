"""Single-window camera wall: a grid that resizes to however many cameras
are configured, with click-to-maximize on a tile and click-to-return."""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np

from src.pipeline.processor import CameraProcessor

logger = logging.getLogger(__name__)

_BG = (30, 30, 30)
_PLACEHOLDER_TEXT_COLOR = (150, 150, 150)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


class GridDisplay:
    WINDOW_NAME = "PPE Monitor"

    def __init__(self, processors: list[CameraProcessor], window_size: tuple[int, int] = (1280, 720)) -> None:
        self.processors = processors
        self.window_w, self.window_h = window_size
        self.focused_index: int | None = None

        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.WINDOW_NAME, self._on_mouse)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    @staticmethod
    def _grid_dims(n: int) -> tuple[int, int]:
        """Columns, rows for an n-tile grid — as square as possible."""
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        return cols, rows

    def _tile_size(self) -> tuple[int, int]:
        cols, rows = self._grid_dims(len(self.processors))
        return self.window_w // cols, self.window_h // rows

    def _placeholder(self, w: int, h: int, text: str) -> np.ndarray:
        img = np.full((h, w, 3), _BG, dtype=np.uint8)
        (tw, th), _ = cv2.getTextSize(text, _FONT, 0.6, 1)
        cv2.putText(img, text, ((w - tw) // 2, (h + th) // 2), _FONT, 0.6, _PLACEHOLDER_TEXT_COLOR, 1, cv2.LINE_AA)
        return img

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self.focused_index is not None:
            self.focused_index = None
            return

        n = len(self.processors)
        if n == 0:
            return
        cols, _ = self._grid_dims(n)
        tile_w, tile_h = self._tile_size()
        col, row = x // tile_w, y // tile_h
        idx = row * cols + col
        if idx < n:
            self.focused_index = idx

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _frame_for(self, proc: CameraProcessor, w: int, h: int) -> np.ndarray:
        frame = proc.get_latest_frame()
        if frame is None:
            return self._placeholder(w, h, f"{proc.camera_name}: connecting...")
        return cv2.resize(frame, (w, h))

    def render(self) -> np.ndarray:
        n = len(self.processors)
        if n == 0:
            return self._placeholder(self.window_w, self.window_h, "No cameras configured")

        if self.focused_index is not None:
            return self._frame_for(self.processors[self.focused_index], self.window_w, self.window_h)

        cols, rows = self._grid_dims(n)
        tile_w, tile_h = self._tile_size()
        canvas = np.full((tile_h * rows, tile_w * cols, 3), _BG, dtype=np.uint8)
        for i, proc in enumerate(self.processors):
            r, c = divmod(i, cols)
            canvas[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = self._frame_for(proc, tile_w, tile_h)
        return canvas

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        try:
            while True:
                cv2.imshow(self.WINDOW_NAME, self.render())
                key = cv2.waitKey(30) & 0xFF
                if key == ord("q"):
                    logger.info("Quit key pressed.")
                    break
                try:
                    if cv2.getWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break
        finally:
            cv2.destroyAllWindows()
