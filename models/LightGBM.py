from ._models import register_model
from functools import partial
from lightgbm import LGBMClassifier

register_model(
    model_name="LightGBM",
    model_description="LightGBM Classifier",
    model_class=partial(
        LGBMClassifier,
        random_state=42,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=-1,
        num_leaves=31,
        min_child_samples=20,
        subsample_for_bin=200000,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=0.0,
        force_row_wise=True
    )
)