from typing import Dict, Any
from .elo import EloModel
from .gbdt_lgbm import LGBMModel

REGISTRY = {
    "elo": EloModel,
    "gbdt": LGBMModel,
}

def make_model(name: str, params: Dict[str, Any]):
    if name not in REGISTRY:
        raise KeyError(f"Unknown model: {name}")
    return REGISTRY[name](**params)
