"""GPU-first model device selection with explicit CPU fallback warnings."""

from __future__ import annotations

import os
from typing import Literal
import warnings

DeviceName = Literal["cuda", "cpu"]


def select_model_device(component: str) -> DeviceName:
    """Prefer CUDA and warn whenever a model will execute on CPU."""
    requested = os.environ.get("MODEL_DEVICE", "auto").strip().lower()
    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError("MODEL_DEVICE must be one of: auto, cuda, cpu")
    if requested == "cpu":
        warnings.warn(
            f"{component}: model execution explicitly configured for CPU",
            RuntimeWarning,
            stacklevel=2,
        )
        return "cpu"

    import torch

    if torch.cuda.is_available():
        return "cuda"
    warnings.warn(
        f"{component}: CUDA is unavailable; falling back to CPU",
        RuntimeWarning,
        stacklevel=2,
    )
    return "cpu"


def model_kwargs_for(device: DeviceName) -> dict[str, object] | None:
    """Use half precision on CUDA and framework defaults on CPU."""
    if device == "cpu":
        return None
    if device != "cuda":
        raise ValueError(f"unsupported model device: {device}")

    import torch

    return {"dtype": torch.float16}
