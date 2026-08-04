"""
CatRange: Protein Kinetic Range Prediction Pipeline

XGBoost-based prediction of enzymatic kinetic parameters (kcat, Km)
using dual-input protein embeddings (ESM-C / ESM-2).

Configuration is driven by YAML files in configs/.
"""

__version__ = "1.2.0"
__all__ = [
    "utils",
    "config",
    "data_pipeline",
    "model_training",
    "evaluation",
]
