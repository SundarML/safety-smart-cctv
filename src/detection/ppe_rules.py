from __future__ import annotations

from dataclasses import dataclass, field

from .detector import Detection


@dataclass
class ViolationReport:
    has_violation: bool
    violations: list[str] = field(default_factory=list)
    violation_detections: list[Detection] = field(default_factory=list)
    all_detections: list[Detection] = field(default_factory=list)


class PPERuleChecker:
    """Checks a list of detections for PPE violations."""

    def __init__(self, violation_classes: list[str]) -> None:
        self.violation_classes: set[str] = set(violation_classes)

    def check(self, detections: list[Detection]) -> ViolationReport:
        violation_detections = [d for d in detections if d.is_violation]
        return ViolationReport(
            has_violation=len(violation_detections) > 0,
            violations=[d.class_name for d in violation_detections],
            violation_detections=violation_detections,
            all_detections=detections,
        )
