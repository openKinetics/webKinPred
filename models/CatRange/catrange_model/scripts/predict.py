#!/usr/bin/env python
"""
Inference script for making predictions on new data.

Usage:
    python scripts/predict.py --model outputs/models/kcat_esmc_fold1.pkl --features embeddings.pt
"""

import argparse
from pathlib import Path

def _load_runtime_dependencies():
    """Import heavy runtime dependencies only after CLI args are parsed."""
    global np, torch, check_device, safe_load, load_model, predict_with_model

    if globals().get("np") is not None:
        return

    import numpy as _np
    import torch as _torch

    from src.utils import check_device as _check_device, safe_load as _safe_load
    from src.model_training import load_model as _load_model, predict_with_model as _predict_with_model

    np = _np
    torch = _torch
    check_device = _check_device
    safe_load = _safe_load
    load_model = _load_model
    predict_with_model = _predict_with_model


def main():
    parser = argparse.ArgumentParser(description="Make predictions with trained CatRange model")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (.pkl)")
    parser.add_argument("--features", type=str, required=True, help="Path to feature tensors (.pt)")
    parser.add_argument("--output", type=str, default="predictions.txt", help="Output file")
    parser.add_argument("--return-proba", action="store_true", help="Return probabilities instead of hard predictions")
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="auto", help="Device to use")
    
    args = parser.parse_args()

    try:
        _load_runtime_dependencies()
    except Exception as exc:
        parser.exit(
            1,
            "CatRange prediction dependencies failed to import. "
            "If this environment was upgraded to NumPy 2.x, repair it with "
            "`python3 -m pip install \"numpy<2\"` and reinstall the package entry points "
            "with `python3 -m pip install --no-deps -e .`.\n"
            f"Original import error: {exc}\n",
        )
    
    print("CatRange Inference")
    print("=" * 50)
    
    # Setup
    if args.device == "auto":
        device = check_device()
    else:
        device = args.device
    print(f"Device: {device}")
    
    # Load model
    print(f"Loading model from: {args.model}")
    model = load_model(args.model)
    print(f"✓ Model loaded")
    
    # Load features
    print(f"Loading features from: {args.features}")
    features = safe_load(args.features, device=device)
    
    if isinstance(features, dict):
        # If saved as dict (data, labels, etc.)
        X = features.get('data', features)
    else:
        X = features
    
    print(f"Features shape: {X.shape}")
    
    # Make predictions
    print(f"Making predictions...")
    if args.return_proba:
        predictions = predict_with_model(model, X, return_proba=True)
        print(f"Predicted probabilities shape: {predictions.shape}")
    else:
        predictions = predict_with_model(model, X, return_proba=False)
        print(f"Predicted classes: {predictions.shape}")
    
    # Save predictions
    if args.return_proba:
        np.savetxt(args.output, predictions, delimiter=",", fmt="%.6f")
        print(f"Saved class probabilities to: {args.output}")
    else:
        np.savetxt(args.output, predictions, delimiter=",", fmt="%d")
        print(f"Saved predictions to: {args.output}")
    
    print(f"\n✓ Inference complete!")


if __name__ == "__main__":
    main()
