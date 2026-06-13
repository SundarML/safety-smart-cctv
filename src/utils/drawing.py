from __future__ import annotations

import cv2
import numpy as np

from src.detection.detector import Detection
from src.detection.ppe_rules import ViolationReport

# BGR colors
_RED = (0, 0, 255)
_GREEN = (0, 200, 0)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_DARK_RED = (0, 0, 180)
_DARK_GREEN = (0, 140, 0)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_STATUS_BAR_H = 40


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        color = _RED if det.is_violation else _GREEN
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{det.class_name} {det.confidence:.0%}"
        (lw, lh), baseline = cv2.getTextSize(label, _FONT, 0.55, 1)
        tag_y1 = max(y1 - lh - baseline - 6, 0)
        cv2.rectangle(frame, (x1, tag_y1), (x1 + lw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - baseline - 2), _FONT, 0.55, _WHITE, 1, cv2.LINE_AA)

    return frame


def draw_status_bar(frame: np.ndarray, camera_name: str, report: ViolationReport) -> np.ndarray:
    h, w = frame.shape[:2]
    bar_color = _DARK_RED if report.has_violation else _DARK_GREEN
    cv2.rectangle(frame, (0, h - _STATUS_BAR_H), (w, h), bar_color, -1)

    if report.has_violation:
        unique_violations = sorted(set(report.violations))
        status = f"VIOLATION: {', '.join(unique_violations)}"
    else:
        status = "ALL CLEAR — PPE Compliant"

    text = f"[{camera_name}]  {status}"
    cv2.putText(frame, text, (10, h - 10), _FONT, 0.65, _WHITE, 2, cv2.LINE_AA)
    return frame
