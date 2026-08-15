"""APR-WM V0: interaction-level physics/residual competition."""

from __future__ import annotations

from typing import Any

__all__ = ["ExperimentConfig", "MODEL_NAMES", "build_model", "load_config"]
__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    if name in {"ExperimentConfig", "load_config"}:
        from .config import ExperimentConfig, load_config

        return {"ExperimentConfig": ExperimentConfig, "load_config": load_config}[name]
    if name in {"MODEL_NAMES", "build_model"}:
        from .models import MODEL_NAMES, build_model

        return {"MODEL_NAMES": MODEL_NAMES, "build_model": build_model}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
