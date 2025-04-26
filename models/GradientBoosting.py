from ._models import register_model
from functools import partial
from sklearn.ensemble import GradientBoostingClassifier

register_model(
    model_name="GradientBoosting",
    model_description="Gradient Boosting Classifier (100 trees, max_depth=3)",
    model_class=partial(
        GradientBoostingClassifier,
        n_estimators=100,        
        learning_rate=0.1,        
        max_depth=3,             
        random_state=42          
    )
)
