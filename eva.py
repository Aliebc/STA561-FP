from models import get_models
import pandas as pd

if __name__ == "__main__":
    models = get_models()
    for model in models:
        print(f"Model Name: {model.model_name}")
        print(f"Model Description: {model.model_description}")
        print(f"Model Class: {model.model_class}")
        ins = model.model_class()
        print(f"Model Instance: {ins}")
        print("-" * 40)