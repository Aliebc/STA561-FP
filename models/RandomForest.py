from functools import partial
from ._models import register_model
from sklearn.ensemble import RandomForestClassifier

register_model(
    model_name="RandomForest",
    model_description="Random Forest Classifier (100 trees, max_depth=10)",
    model_class=partial(
        RandomForestClassifier,
        n_estimators=100,        
        max_depth=10,            
        random_state=42,         
        class_weight="balanced"  
    )
)

