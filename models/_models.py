from typing import List, Dict, Type, TypedDict
#from sklearn.base import Classifier
from dataclasses import dataclass
from typing import Protocol, TypeVar, Any
import numpy as np

T = TypeVar("T", bound="Classifier")

class Classifier(Protocol):
    def fit(self, X: Any, y: Any) -> "Classifier": ...
    def predict(self, X: Any) -> np.ndarray: ...
    def predict_proba(self, X: Any) -> np.ndarray: ...

@dataclass
class ModelEntry():
    model_name: str
    model_description: str
    model_class: Type[Classifier]

_models: List[ModelEntry] = []

def register_model(
    model_name: str,
    model_description: str,
    model_class: Type[Classifier],
) -> Type[Classifier]:
    _models.append(ModelEntry(model_name, model_description, model_class))
    return model_class

def get_models() -> List[ModelEntry]:
    return _models

def find_model(model_name: str) -> ModelEntry:
    for model in _models:
        if model.model_name == model_name:
            return model
    raise ValueError(f"Model {model_name} not found.")