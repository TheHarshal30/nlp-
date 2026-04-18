import importlib.util
import inspect
import sys
from pathlib import Path

from .base import BaseEmbedder


def _apply_inference_options(embedder, inference: dict | None):
    inference = inference or {}
    if not inference:
        return embedder

    mode = inference.get("inference_mode")
    use_projection = inference.get("use_projection")
    if use_projection is None and mode is not None:
        use_projection = mode == "projected"

    if hasattr(embedder, "metadata") and isinstance(getattr(embedder, "metadata", None), dict):
        if use_projection is not None:
            embedder.metadata["use_projection_at_inference"] = bool(use_projection)
        if mode is not None:
            embedder.metadata["inference_mode"] = mode
    setattr(embedder, "inference_mode", mode or ("projected" if use_projection else "base"))
    setattr(embedder, "use_projection", bool(use_projection))
    return embedder


def load_embedder(model_path: str | Path, inference: dict | None = None):
    model_path = Path(model_path)
    model_file = model_path / "model.py"
    if not model_file.exists():
        raise FileNotFoundError(f"missing model.py: {model_file}")

    base_module = sys.modules.get("base_embedder")
    sys.modules["base_embedder"] = sys.modules[__name__]
    setattr(sys.modules[__name__], "BaseEmbedder", BaseEmbedder)
    try:
        spec = importlib.util.spec_from_file_location(f"embedder_{model_path.name}", model_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"unable to import {model_file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if base_module is not None:
            sys.modules["base_embedder"] = base_module
        else:
            sys.modules.pop("base_embedder", None)

    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ != module.__name__:
            continue
        if hasattr(obj, "load") and hasattr(obj, "encode"):
            embedder = obj()
            embedder.load(str(model_path))
            return _apply_inference_options(embedder, inference)
    raise RuntimeError(f"no embedder class found in {model_file}")
