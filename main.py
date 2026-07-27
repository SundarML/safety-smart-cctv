"""Entry point for the PPE Safety Monitoring System.

Usage examples:
    # Run with config file (default):
    python main.py

    # Quick test on your webcam:
    python main.py --source 0

    # Run on a video file:
    python main.py --source path/to/video.mp4

    # Run on an RTSP stream:
    python main.py --source "rtsp://user:pass@192.168.1.100:554/stream"

    # Use a different config file:
    python main.py --config config/settings.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import threading

import yaml

from src.pipeline.display import GridDisplay
from src.pipeline.processor import CameraProcessor
from src.utils.dvr_discovery import resolve_dvr_hosts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_ENV_VAR_RE = re.compile(r"\$\{(\w+)\}")


def _load_dotenv(path: str = ".env") -> None:
    """Populate os.environ from a .env file, without overriding real env vars."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def _resolve_env_vars(value: str) -> str:
    """Substitute ${VAR_NAME} placeholders (e.g. in RTSP credentials) from the environment."""

    def sub(match: re.Match) -> str:
        var = match.group(1)
        resolved = os.environ.get(var)
        if resolved is None:
            raise RuntimeError(
                f"Config references ${{{var}}} but that environment variable is not set "
                f"(add it to .env — see .env.example)"
            )
        return resolved

    return _ENV_VAR_RE.sub(sub, value)


def load_config(path: str) -> dict:
    _load_dotenv()
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for cam in cfg.get("cameras", []):
        if isinstance(cam.get("source"), str):
            cam["source"] = _resolve_env_vars(cam["source"])
    resolve_dvr_hosts(cfg.get("cameras", []), config_path=path)
    return cfg


def build_processor(cfg: dict, camera_cfg: dict) -> CameraProcessor:
    return CameraProcessor(
        camera_cfg=camera_cfg,
        model_cfg=cfg["model"],
        ppe_cfg=cfg["ppe"],
        display_cfg=cfg["display"],
        log_cfg=cfg["logging"],
        fire_smoke_cfg=cfg.get("fire_smoke"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="PPE Safety Monitoring System")
    parser.add_argument("--config", default="config/settings.yaml", help="Path to YAML config file")
    parser.add_argument("--source", help="Override camera source (webcam index, file path, or RTSP URL)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.source:
        # Single-camera override — try to coerce to int for webcam indices
        try:
            source = int(args.source)
        except ValueError:
            source = args.source
        cameras = [{"id": "cli", "name": "CLI Source", "source": source}]
    else:
        cameras = cfg["cameras"]
        # Coerce webcam indices from YAML (parsed as int already, but be safe)
        for cam in cameras:
            if isinstance(cam["source"], str) and cam["source"].isdigit():
                cam["source"] = int(cam["source"])

    logger.info("Starting PPE monitoring — %d camera(s)", len(cameras))

    processors = [build_processor(cfg, cam) for cam in cameras]
    threads = [
        threading.Thread(target=proc.run, daemon=True, name=cam["id"])
        for proc, cam in zip(processors, cameras)
    ]
    for t in threads:
        t.start()

    if cfg["display"].get("show_window", True):
        display_cfg = cfg["display"]
        GridDisplay(
            processors,
            window_size=(display_cfg.get("frame_width", 1280), display_cfg.get("frame_height", 720)),
        ).run()

    for proc in processors:
        proc.request_stop()
    for t in threads:
        t.join(timeout=5.0)


if __name__ == "__main__":
    main()
