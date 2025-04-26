from ._models import register_model
from functools import partial
from catboost import CatBoostClassifier

register_model(
    model_name="CatBoost",
    model_description="CatBoost Classifier (100 trees, depth=6)",
    model_class=partial(
        CatBoostClassifier,
        iterations=100,        
        learning_rate=0.1,     
        depth=6,               
        verbose=0,             
        random_seed=42         
    )
)
