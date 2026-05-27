"""Resource files for Sched-IR."""

from pathlib import Path

RESOURCE_DIR = Path(__file__).resolve().parent
DA4ML_RESOURCE_YAML = RESOURCE_DIR / "da4ml_resource.yaml"

__all__ = ["DA4ML_RESOURCE_YAML", "RESOURCE_DIR"]
