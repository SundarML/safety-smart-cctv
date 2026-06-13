"""Download a PPE dataset from Roboflow Universe and train a YOLOv8 model.

Target dataset (public, free):
  Roboflow Universe — Construction Site Safety
  https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety
  Classes: Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest,
           Person, Safety Cone, Safety Vest, machinery, vehicle

Quick start:
    # Set your Roboflow API key once (get it at roboflow.com → Settings → API Keys):
    set ROBOFLOW_API_KEY=your_key_here          # Windows CMD
    $env:ROBOFLOW_API_KEY="your_key_here"       # PowerShell

    # Run with defaults (CPU, 50 epochs, nano model):
    python scripts/train.py

    # Run on GPU, more epochs, larger model:
    python scripts/train.py --device cuda --epochs 100 --model yolov8s.pt

    # Use a different Roboflow project:
    python scripts/train.py --workspace myworkspace --project myproject --version 1
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default dataset: Construction Site Safety (Roboflow Universe, public)
# ---------------------------------------------------------------------------
DEFAULT_WORKSPACE = "roboflow-universe-projects"
DEFAULT_PROJECT   = "construction-site-safety"
DEFAULT_VERSION   = 30          # latest stable version as of mid-2025

MODELS_DIR = Path("models")
RUNS_DIR   = Path("runs")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download PPE dataset and train YOLOv8")

    # Roboflow
    p.add_argument("--api-key",   default=os.getenv("ROBOFLOW_API_KEY"), help="Roboflow private API key (or set ROBOFLOW_API_KEY env var)")
    p.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    p.add_argument("--project",   default=DEFAULT_PROJECT)
    p.add_argument("--version",   default=DEFAULT_VERSION, type=int)

    # Model
    p.add_argument("--model",   default="yolov8n.pt", help="Base YOLOv8 weights to start from (yolov8n/s/m/l/x.pt)")
    p.add_argument("--epochs",  default=50,  type=int)
    p.add_argument("--batch",   default=16,  type=int,   help="Batch size (-1 = auto)")
    p.add_argument("--imgsz",   default=640, type=int,   help="Training image size")
    p.add_argument("--device",  default="cpu",           help="'cpu', 'cuda', '0', '0,1', etc.")
    p.add_argument("--patience", default=20, type=int,   help="Early stopping patience (epochs)")

    # Output
    p.add_argument("--output",  default=str(MODELS_DIR / "ppe.pt"), help="Where to copy the best weights after training")
    p.add_argument("--skip-download", action="store_true", help="Skip download if dataset folder already exists")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Step 1 — Download
# ---------------------------------------------------------------------------

def download_dataset(args: argparse.Namespace) -> Path:
    if not args.api_key:
        logger.error(
            "No Roboflow API key found.\n"
            "  1. Sign up free at https://roboflow.com\n"
            "  2. Go to Settings → API Keys → copy your Private API key\n"
            "  3. Set it:  $env:ROBOFLOW_API_KEY='your_key'  (PowerShell)\n"
            "              set ROBOFLOW_API_KEY=your_key      (CMD)\n"
            "  4. Re-run this script."
        )
        sys.exit(1)

    try:
        from roboflow import Roboflow
    except ImportError:
        logger.error("roboflow package not installed. Run: pip install roboflow")
        sys.exit(1)

    dataset_dir = Path("datasets") / f"{args.project}-{args.version}"

    if args.skip_download and dataset_dir.exists():
        logger.info("Skipping download — dataset folder already exists: %s", dataset_dir)
        return dataset_dir

    logger.info("Connecting to Roboflow (%s / %s / v%d)…", args.workspace, args.project, args.version)
    rf = Roboflow(api_key=args.api_key)
    project = rf.workspace(args.workspace).project(args.project)
    dataset = project.version(args.version).download("yolov8", location=str(dataset_dir))
    logger.info("Dataset downloaded to: %s", dataset.location)
    return Path(dataset.location)


# ---------------------------------------------------------------------------
# Step 2 — Train
# ---------------------------------------------------------------------------

def train(dataset_dir: Path, args: argparse.Namespace) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics package not installed. Run: pip install ultralytics")
        sys.exit(1)

    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        logger.error("data.yaml not found at %s — dataset may not have downloaded correctly.", data_yaml)
        sys.exit(1)

    logger.info("Starting training: model=%s  epochs=%d  device=%s  imgsz=%d", args.model, args.epochs, args.device, args.imgsz)

    model = YOLO(args.model)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        patience=args.patience,
        project=str(RUNS_DIR),
        name="ppe_training",
        save=True,
        plots=True,
        verbose=True,
    )

    # ultralytics saves best weights to runs/ppe_training/weights/best.pt
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if not best_weights.exists():
        logger.error("best.pt not found at %s", best_weights)
        sys.exit(1)

    logger.info("Training complete. Best weights: %s", best_weights)
    return best_weights


# ---------------------------------------------------------------------------
# Step 3 — Export to models/ppe.pt
# ---------------------------------------------------------------------------

def export_weights(best_weights: Path, output: str) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, output_path)
    logger.info("Model saved to: %s  (update config/settings.yaml if path differs)", output_path)


# ---------------------------------------------------------------------------
# Step 4 — Print the detected class names so user can verify settings.yaml
# ---------------------------------------------------------------------------

def print_class_summary(dataset_dir: Path) -> None:
    import yaml
    data_yaml = dataset_dir / "data.yaml"
    with open(data_yaml) as f:
        meta = yaml.safe_load(f)
    names: list[str] = meta.get("names", [])
    logger.info("Dataset classes (%d total):", len(names))
    for i, name in enumerate(names):
        logger.info("  %2d: %s", i, name)
    violation_candidates = [n for n in names if n.upper().startswith("NO-") or "NO " in n.upper()]
    if violation_candidates:
        logger.info("Suggested violation_classes for config/settings.yaml: %s", violation_candidates)


# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    MODELS_DIR.mkdir(exist_ok=True)
    RUNS_DIR.mkdir(exist_ok=True)

    dataset_dir = download_dataset(args)
    print_class_summary(dataset_dir)
    best_weights = train(dataset_dir, args)
    export_weights(best_weights, args.output)

    logger.info("")
    logger.info("All done! Next steps:")
    logger.info("  1. Verify 'violation_classes' in config/settings.yaml match the class names above.")
    logger.info("  2. Run:  python main.py --source 0")


if __name__ == "__main__":
    main()
