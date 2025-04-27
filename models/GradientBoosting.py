from ._models import register_model
from functools import partial
from sklearn.ensemble import GradientBoostingClassifier

register_model(
    model_name="GradientBoosting",
    model_description="Gradient Boosting Classifier (100 trees, max_depth=3)",
    model_class=partial(
        GradientBoostingClassifier,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=2,
        subsample=0.8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
)
