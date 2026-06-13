from __future__ import annotations

"""Alert notifier — placeholder for email / SMS integration."""

import logging
import smtplib
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)


class EmailNotifier:
    """Send a plain-text email alert on violation.

    Configure via config/settings.yaml (alerts section) once you are ready
    to wire up live notifications.

    Example config block:
        alerts:
          email:
            enabled: false
            smtp_host: smtp.gmail.com
            smtp_port: 587
            sender: you@example.com
            password: "app-password"
            recipients:
              - supervisor@example.com
    """

    def __init__(self, cfg: dict) -> None:
        self.enabled: bool = cfg.get("enabled", False)
        self.smtp_host: str = cfg.get("smtp_host", "")
        self.smtp_port: int = cfg.get("smtp_port", 587)
        self.sender: str = cfg.get("sender", "")
        self.password: str = cfg.get("password", "")
        self.recipients: list[str] = cfg.get("recipients", [])

    def send(self, camera_name: str, violations: list[str]) -> None:
        if not self.enabled:
            return
        subject = f"[PPE Alert] Violation detected — {camera_name}"
        body = f"Camera: {camera_name}\nViolations: {', '.join(violations)}"
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = self.sender
        msg["To"] = ", ".join(self.recipients)
        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender, self.password)
                server.sendmail(self.sender, self.recipients, msg.as_string())
            logger.info("Alert email sent for %s", camera_name)
        except Exception as exc:
            logger.error("Failed to send alert email: %s", exc)
