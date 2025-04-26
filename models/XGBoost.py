from ._models import register_model
from functools import partial
from xgboost import XGBClassifier

register_model(
    model_name="XGBoost",
    model_description="XGBoost Classifier (100 trees, max_depth=6)",
    model_class=partial(
        XGBClassifier,
        n_estimators=100,       
        learning_rate=0.1,      
        max_depth=6,            
        use_label_encoder=False,
        eval_metric='mlogloss', 
        random_state=42         
    )
)

