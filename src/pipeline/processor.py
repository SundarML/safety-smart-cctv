from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

import cv2

from src.detection.detector import PPEDetector
from src.detection.ppe_rules import PPERuleChecker, ViolationReport
from src.pipeline.stream import VideoStream
from src.utils.drawing import draw_detections, draw_status_bar

logger = logging.getLogger(__name__)


class CameraProcessor:
    """
    Orchestrates one camera feed end-to-end:
      VideoStream → PPEDetector → PPERuleChecker → display / logging
    """

    # Minimum seconds between saving frames for the same camera to avoid
    # flooding the disk when a worker stays in violation.
    _SAVE_COOLDOWN = 5.0

    def __init__(
        self,
        camera_cfg: dict,
        model_cfg: dict,
        ppe_cfg: dict,
        display_cfg: dict,
        log_cfg: dict,
    ) -> None:
        self.camera_id: str = camera_cfg["id"]
        self.camera_name: str = camera_cfg.get("name", self.camera_id)
        source = camera_cfg["source"]

        self.stream = VideoStream(
            source=source,
            width=display_cfg.get("frame_width"),
            height=display_cfg.get("frame_height"),
        )
        self.detector = PPEDetector(
            weights=model_cfg["weights"],
            confidence=model_cfg["confidence"],
            iou=model_cfg["iou"],
            device=model_cfg["device"],
            violation_classes=ppe_cfg["violation_classes"],
        )
        self.rule_checker = PPERuleChecker(violation_classes=ppe_cfg["violation_classes"])

        self.show_window: bool = display_cfg.get("show_window", True)
        self.save_violations: bool = log_cfg.get("save_violations", True)
        self.save_frames: bool = log_cfg.get("save_frames", True)

        self.log_dir = Path(log_cfg.get("log_dir", "logs")) / self.camera_id
        if self.save_violations:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._init_csv_log()

        self._last_save_time = 0.0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        self.stream.start()
        logger.info("[%s] Stream started (source=%r)", self.camera_name, self.stream.source)

        try:
            while self.stream.is_running():
                ret, frame = self.stream.read()
                if not ret:
                    continue

                detections = self.detector.detect(frame)
                report = self.rule_checker.check(detections)

                annotated = draw_detections(frame.copy(), detections)
                annotated = draw_status_bar(annotated, self.camera_name, report)

                if report.has_violation:
                    self._handle_violation(frame, report)

                if self.show_window:
                    cv2.imshow(f"PPE Monitor — {self.camera_name}", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("[%s] Quit key pressed.", self.camera_name)
                        break
        finally:
            self.stream.stop()
            cv2.destroyAllWindows()
            logger.info("[%s] Stream stopped.", self.camera_name)

    # ------------------------------------------------------------------
    # Violation handling
    # ------------------------------------------------------------------

    def _handle_violation(self, frame, report: ViolationReport) -> None:
        if not self.save_violations:
            return

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        unique_violations = sorted(set(report.violations))

        # Append to CSV log
        with open(self.log_dir / "violations.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, self.camera_name, "; ".join(unique_violations)])

        # Save snapshot with cooldown so one prolonged event doesn't flood disk
        now = time.monotonic()
        if self.save_frames and (now - self._last_save_time) >= self._SAVE_COOLDOWN:
            ts_file = time.strftime("%Y%m%d_%H%M%S")
            path = self.log_dir / f"violation_{ts_file}.jpg"
            cv2.imwrite(str(path), frame)
            logger.warning("[%s] Violation saved: %s → %s", self.camera_name, unique_violations, path)
            self._last_save_time = now

    def _init_csv_log(self) -> None:
        csv_path = self.log_dir / "violations.csv"
        if not csv_path.exists():
            with open(csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["timestamp", "camera", "violations"])
