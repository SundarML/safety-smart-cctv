from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    is_violation: bool = False


class PPEDetector:
    def __init__(
        self,
        weights: str,
        confidence: float,
        iou: float,
        device: str,
        violation_classes: list[str],
    ) -> None:
        weights_path = Path(weights)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found: {weights}\n"
                "Download a PPE model from Roboflow Universe and place the .pt file "
                f"at '{weights}', or update 'model.weights' in config/settings.yaml."
            )
        self.model = YOLO(str(weights_path))
        self.confidence = confidence
        self.iou = iou
        self.device = device
        self.violation_classes: set[str] = set(violation_classes)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model(
            frame,
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(
                    Detection(
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        is_violation=cls_name in self.violation_classes,
                    )
                )
        return detections

    @property
    def class_names(self) -> dict[int, str]:
        return self.model.names
