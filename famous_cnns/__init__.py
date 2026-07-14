"""High-level API for the from-scratch CNN implementations in this project."""

from .factory import ModelInfo, adapt_input_channels, create_model, list_models, model_info
from .optimizers import create_optimizer
from .orchestrator import CNNOrchestrator

__all__ = [
    "CNNOrchestrator",
    "ModelInfo",
    "adapt_input_channels",
    "create_model",
    "create_optimizer",
    "list_models",
    "model_info",
]

__version__ = "0.3.0"
