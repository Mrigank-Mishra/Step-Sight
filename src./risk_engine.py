"""Foothold Safety Assistant - Terrain Risk Engine.

Computes a 12x16 float32 risk grid in [0, 1] for an input BGR frame:
- Below 0.35: SAFE (green)
- 0.35 to 0.60: UNCERTAIN (yellow)
- Above 0.60: RISKY (red)

This module is designed to be swappable:
- 'heuristic': Image rule-based detection of cracks, glare, and slick mud (fast, runs on CPU).
- 'model': Machine learning model (e.g. YOLOv8 classifier / segmenter, planned for Phase 5).
"""
from typing import Any, Optional

import cv2
import numpy as np

# Processing resolution and risk grid dimensions (must divide evenly)
PROC_W, PROC_H = 320, 240
GRID_ROWS, GRID_COLS = 12, 16
assert PROC_H % GRID_ROWS == 0 and PROC_W % GRID_COLS == 0, "Grid must divide processing size evenly"

# Hazard weights for heuristic detection (sums to 1.0)
W_CRACK = 0.45  # Broken / cracked / loose gravel (high edge density)
W_GLARE = 0.35  # Shiny specular highlights (wet rock, ice)
W_SLICK = 0.20  # Dark, low-texture areas (mud, slick wet patches)

# Risk thresholds
SAFE_THRESHOLD = 0.35
RISK_RED = 0.60


def cell_mean(mask: np.ndarray, rows: int = GRID_ROWS, cols: int = GRID_COLS) -> np.ndarray:
    """Average a 2D mask inside each grid cell -> output shape (rows, cols).

    Args:
        mask: 2D array of shape (H, W).
        rows: Target number of grid rows.
        cols: Target number of grid columns.

    Returns:
        2D float array of shape (rows, cols) containing cell-wise means.
    """
    h, w = mask.shape
    return mask.reshape(rows, h // rows, cols, w // cols).mean(axis=(1, 3))


def compute_heuristic_risk_grid(frame: np.ndarray, sensitivity: float = 1.3) -> np.ndarray:
    """Compute risk grid using visual heuristics (cracks, specular glare, dark slick mud).

    Args:
        frame: BGR video or image frame.
        sensitivity: Sensitivity multiplier (>1.0 leans more cautious).

    Returns:
        float32 array of shape (GRID_ROWS, GRID_COLS) with values in [0, 1].
    """
    # Downscale for real-time speed (320x240)
    small = cv2.resize(frame, (PROC_W, PROC_H))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 1. Crack / edge cue: lots of edges indicate broken, uneven, or loose terrain
    edges = (cv2.Canny(blur, 60, 140) > 0).astype(np.float32)

    # 2. Glare cue: high value (brightness) with low saturation indicates slippery wet rock
    glare = ((hsv[..., 2] > 225) & (hsv[..., 1] < 50)).astype(np.float32)

    # 3. Slick cue: low brightness and low Laplacian variance indicates dark smooth mud
    lap = np.abs(cv2.Laplacian(blur, cv2.CV_32F))
    slick = ((hsv[..., 2] < 80) & (lap < 2.0)).astype(np.float32)

    # Compute normalized cell scores in [0, 1]
    crack_score = np.clip(cell_mean(edges, GRID_ROWS, GRID_COLS) / 0.25, 0.0, 1.0)
    glare_score = np.clip(cell_mean(glare, GRID_ROWS, GRID_COLS) / 0.10, 0.0, 1.0)
    slick_score = np.clip(cell_mean(slick, GRID_ROWS, GRID_COLS) / 0.50, 0.0, 1.0)

    # Weighted risk aggregation
    risk = W_CRACK * crack_score + W_GLARE * glare_score + W_SLICK * slick_score
    return np.clip(risk * sensitivity, 0.0, 1.0).astype(np.float32)


_CACHED_MODEL = None
_CACHED_PATH = None
_CACHED_MAPPINGS = None


def load_model(weights_path: str = "models/best.pt") -> Any:
    """Load and cache the trained YOLO classification model."""
    global _CACHED_MODEL, _CACHED_PATH, _CACHED_MAPPINGS

    import sys
    from pathlib import Path
    
    if getattr(sys, "frozen", False):
        wpath = Path(sys._MEIPASS) / weights_path
    else:
        # relative to project root
        wpath = Path(__file__).resolve().parent.parent / weights_path

    if not wpath.exists():
        raise FileNotFoundError(
            f"Trained model weights not found at '{wpath.resolve()}'.\n"
            "Please run model training first:\n"
            "  python training/train_yolo.py"
        )

    if _CACHED_MODEL is not None and _CACHED_PATH == str(wpath.resolve()):
        return _CACHED_MODEL

    from ultralytics import YOLO
    model = YOLO(str(wpath))
    _CACHED_MODEL = model
    _CACHED_PATH = str(wpath.resolve())

    # Map class names to indices
    # Model names dictionary: e.g. {0: 'risky', 1: 'safe', 2: 'uncertain'}
    name_to_idx = {name.lower(): idx for idx, name in model.names.items()}
    _CACHED_MAPPINGS = (
        name_to_idx.get("safe", 0),
        name_to_idx.get("uncertain", 1),
        name_to_idx.get("risky", 2),
    )
    return _CACHED_MODEL


def compute_model_risk_grid(
    frame: np.ndarray,
    model: Optional[Any] = None,
    sensitivity: float = 1.3,
) -> np.ndarray:
    """Compute (12, 16) risk grid using the trained YOLOv8 classification model.

    Evaluates all 192 grid patches simultaneously in one batched forward pass on CPU.
    Risk mapping:
      risk = 0.0 * P(safe) + 0.5 * P(uncertain) + 1.0 * P(risky)

    Args:
        frame: BGR video or image frame.
        model: Optional pre-loaded YOLO model or weights path.
        sensitivity: Sensitivity multiplier.

    Returns:
        float32 array of shape (GRID_ROWS, GRID_COLS) with values in [0, 1].
    """
    global _CACHED_MAPPINGS
    if model is None:
        model = load_model("models/best.pt")
    elif isinstance(model, str):
        model = load_model(model)

    if _CACHED_MAPPINGS is None:
        name_to_idx = {name.lower(): idx for idx, name in model.names.items()}
        _CACHED_MAPPINGS = (
            name_to_idx.get("safe", 0),
            name_to_idx.get("uncertain", 1),
            name_to_idx.get("risky", 2),
        )

    idx_safe, idx_unc, idx_risk = _CACHED_MAPPINGS

    # 1. Resize frame to processing resolution (320x240) and convert BGR -> RGB
    small = cv2.resize(frame, (PROC_W, PROC_H))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    # 2. Vectorized extraction of all 192 patches (20x20 each)
    # Shape: (12, 20, 16, 20, 3) -> (192, 3, 20, 20)
    patches = (
        rgb.reshape(GRID_ROWS, PROC_H // GRID_ROWS, GRID_COLS, PROC_W // GRID_COLS, 3)
        .transpose(0, 2, 4, 1, 3)
        .reshape(GRID_ROWS * GRID_COLS, 3, PROC_H // GRID_ROWS, PROC_W // GRID_COLS)
    )

    import torch
    import torch.nn.functional as F

    tensor_patches = torch.from_numpy(patches).float() / 255.0
    # Upsample to 64x64 matching model training image size
    batch = F.interpolate(tensor_patches, size=(64, 64), mode="bilinear", align_corners=False)

    # 3. Model inference
    pytorch_net = model.model
    pytorch_net.eval()

    with torch.no_grad():
        logits = pytorch_net(batch)
        probs = torch.softmax(logits, dim=1)

    # 4. Compute composite risk per cell: 0.0 for safe, 0.5 for uncertain, 1.0 for risky
    p_safe = probs[:, idx_safe].numpy()
    p_unc = probs[:, idx_unc].numpy()
    p_risk = probs[:, idx_risk].numpy()

    cell_risk = 0.0 * p_safe + 0.5 * p_unc + 1.0 * p_risk
    cell_risk = np.clip(cell_risk * sensitivity, 0.0, 1.0)

    # Reshape back to (GRID_ROWS, GRID_COLS)
    grid = cell_risk.reshape(GRID_ROWS, GRID_COLS).astype(np.float32)
    return grid


def compute_risk_grid(
    frame: np.ndarray,
    engine: str = "heuristic",
    sensitivity: float = 1.3,
    model: Optional[Any] = None,
) -> np.ndarray:
    """Swappable interface for computing the terrain risk grid.

    Args:
        frame: Input BGR frame.
        engine: 'heuristic' (default) or 'model'.
        sensitivity: Risk sensitivity multiplier.
        model: Optional model instance if engine='model'.

    Returns:
        float32 array of shape (GRID_ROWS, GRID_COLS) in [0, 1].
    """
    engine_normalized = engine.strip().lower()
    if engine_normalized == "heuristic":
        return compute_heuristic_risk_grid(frame, sensitivity=sensitivity)
    elif engine_normalized == "model":
        return compute_model_risk_grid(frame, model=model, sensitivity=sensitivity)
    else:
        raise ValueError(
            f"Unknown risk engine '{engine}'. Supported options: 'heuristic', 'model'."
        )
