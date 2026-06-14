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
import threading

import yaml

from src.pipeline.processor import CameraProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_camera(cfg: dict, camera_cfg: dict) -> None:
    processor = CameraProcessor(
        camera_cfg=camera_cfg,
        model_cfg=cfg["model"],
        ppe_cfg=cfg["ppe"],
        display_cfg=cfg["display"],
        log_cfg=cfg["logging"],
        fire_smoke_cfg=cfg.get("fire_smoke"),
    )
    processor.run()


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

    if len(cameras) == 1:
        run_camera(cfg, cameras[0])
    else:
        threads = [
            threading.Thread(target=run_camera, args=(cfg, cam), daemon=True, name=cam["id"])
            for cam in cameras
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()


if __name__ == "__main__":
    main()
