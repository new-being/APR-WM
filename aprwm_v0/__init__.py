"""APR-WM V0: interaction-level physics/residual competition."""

from .config import ExperimentConfig, load_config
from .models import MODEL_NAMES, build_model

__all__ = ["ExperimentConfig", "MODEL_NAMES", "build_model", "load_config"]
__version__ = "0.1.0"

