from __future__ import annotations

import csv
import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from src.detection.detector import PPEDetector
from src.detection.ppe_rules import PPERuleChecker, ViolationReport
from src.pipeline.stream import VideoStream
from src.utils.drawing import draw_detections, draw_hazard_detections, draw_status_bar

logger = logging.getLogger(__name__)


class CameraProcessor:
    """
    Orchestrates one camera feed end-to-end:
      VideoStream → PPEDetector(s) → PPERuleChecker(s) → display / logging
    """

    _SAVE_COOLDOWN = 5.0

    def __init__(
        self,
        camera_cfg: dict,
        model_cfg: dict,
        ppe_cfg: dict,
        display_cfg: dict,
        log_cfg: dict,
        fire_smoke_cfg: dict | None = None,
    ) -> None:
        self.camera_id: str = camera_cfg["id"]
        self.camera_name: str = camera_cfg.get("name", self.camera_id)
        source = camera_cfg["source"]

        self.stream = VideoStream(
            source=source,
            width=display_cfg.get("frame_width"),
            height=display_cfg.get("frame_height"),
        )

        # PPE detector (optional)
        self.ppe_enabled: bool = model_cfg.get("enabled", True)
        self.ppe_detector: PPEDetector | None = None
        self.ppe_checker: PPERuleChecker | None = None
        if self.ppe_enabled:
            self.ppe_detector = PPEDetector(
                weights=model_cfg["weights"],
                confidence=model_cfg["confidence"],
                iou=model_cfg["iou"],
                device=model_cfg["device"],
                violation_classes=ppe_cfg["violation_classes"],
            )
            self.ppe_checker = PPERuleChecker(violation_classes=ppe_cfg["violation_classes"])

        # Fire/smoke detector (optional)
        self.fire_smoke_enabled: bool = bool(
            fire_smoke_cfg and fire_smoke_cfg.get("enabled", True)
        )
        self.fire_smoke_detector: PPEDetector | None = None
        self.fire_smoke_checker: PPERuleChecker | None = None
        if self.fire_smoke_enabled and fire_smoke_cfg:
            alert_classes = fire_smoke_cfg.get("alert_classes", ["fire", "smoke"])
            self.fire_smoke_detector = PPEDetector(
                weights=fire_smoke_cfg["weights"],
                confidence=fire_smoke_cfg["confidence"],
                iou=fire_smoke_cfg["iou"],
                device=fire_smoke_cfg["device"],
                violation_classes=alert_classes,
            )
            self.fire_smoke_checker = PPERuleChecker(violation_classes=alert_classes)

        self._frame_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._stop_requested = threading.Event()

        self.save_violations: bool = log_cfg.get("save_violations", True)
        self.save_frames: bool = log_cfg.get("save_frames", True)

        self.log_dir = Path(log_cfg.get("log_dir", "logs")) / self.camera_id
        if self.save_violations:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._init_csv_log("violations.csv", ["timestamp", "camera", "violations"])
            if self.fire_smoke_enabled:
                self._init_csv_log("hazards.csv", ["timestamp", "camera", "hazards"])

        self._last_ppe_save   = 0.0
        self._last_hazard_save = 0.0

        active = []
        if self.ppe_enabled:
            active.append("PPE")
        if self.fire_smoke_enabled:
            active.append("Fire/Smoke")
        logger.info("[%s] Active detectors: %s", self.camera_name, ", ".join(active) or "none")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        self.stream.start()
        logger.info("[%s] Stream started (source=%r)", self.camera_name, self.stream.source)

        try:
            while self.stream.is_running() and not self._stop_requested.is_set():
                ret, frame = self.stream.read()
                if not ret:
                    continue

                annotated = frame.copy()
                ppe_report: ViolationReport | None = None
                fire_smoke_report: ViolationReport | None = None

                if self.ppe_enabled and self.ppe_detector and self.ppe_checker:
                    ppe_detections = self.ppe_detector.detect(frame)
                    ppe_report = self.ppe_checker.check(ppe_detections)
                    annotated = draw_detections(annotated, ppe_detections)
                    if ppe_report.has_violation:
                        self._log_event(frame, ppe_report.violations, "violations.csv", "violation", "_last_ppe_save")

                if self.fire_smoke_enabled and self.fire_smoke_detector and self.fire_smoke_checker:
                    fs_detections = self.fire_smoke_detector.detect(frame)
                    fire_smoke_report = self.fire_smoke_checker.check(fs_detections)
                    annotated = draw_hazard_detections(annotated, fs_detections)
                    if fire_smoke_report.has_violation:
                        self._log_event(frame, fire_smoke_report.violations, "hazards.csv", "hazards", "_last_hazard_save")

                annotated = draw_status_bar(annotated, self.camera_name, ppe_report, fire_smoke_report)

                with self._frame_lock:
                    self._latest_frame = annotated
        finally:
            self.stream.stop()
            logger.info("[%s] Stream stopped.", self.camera_name)

    def get_latest_frame(self) -> np.ndarray | None:
        """Most recently annotated frame, or None if the stream hasn't produced one yet."""
        with self._frame_lock:
            return self._latest_frame

    def request_stop(self) -> None:
        self._stop_requested.set()

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log_event(
        self,
        frame,
        labels: list[str],
        csv_filename: str,
        frame_prefix: str,
        cooldown_attr: str,
    ) -> None:
        if not self.save_violations:
            return

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        unique_labels = sorted(set(labels))

        with open(self.log_dir / csv_filename, "a", newline="") as f:
            csv.writer(f).writerow([timestamp, self.camera_name, "; ".join(unique_labels)])

        now = time.monotonic()
        if self.save_frames and (now - getattr(self, cooldown_attr)) >= self._SAVE_COOLDOWN:
            ts_file = time.strftime("%Y%m%d_%H%M%S")
            path = self.log_dir / f"{frame_prefix}_{ts_file}.jpg"
            cv2.imwrite(str(path), frame)
            logger.warning("[%s] %s saved → %s", self.camera_name, unique_labels, path)
            setattr(self, cooldown_attr, now)

    def _init_csv_log(self, filename: str, headers: list[str]) -> None:
        path = self.log_dir / filename
        if not path.exists():
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(headers)
