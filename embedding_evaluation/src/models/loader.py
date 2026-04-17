import importlib.util
import inspect
import sys
from pathlib import Path

from .base import BaseEmbedder


def load_embedder(model_path: str | Path):
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
            return embedder
    raise RuntimeError(f"no embedder class found in {model_file}")
