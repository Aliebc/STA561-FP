import importlib
import os
import glob

from ._models import get_models, find_model

__all__ = [
    "get_models",
    "find_model",
]

current_dir = os.path.dirname(__file__)
module_files = glob.glob(os.path.join(current_dir, "[!_]*.py"))

for file_path in module_files:
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    full_module_name = f"{__name__}.{module_name}"
    importlib.import_module(full_module_name)
