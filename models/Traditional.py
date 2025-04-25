from ._models import register_model
from functools import partial
from sklearn.linear_model import (
    LogisticRegression,
    LogisticRegressionCV,
)

register_model(
    model_name="Logistic",
    model_description="Logistic Regression Classifier",
    model_class=partial(
        LogisticRegression,
        max_iter=20000,
        penalty='l2',
        random_state=42,
    )
)

register_model(
    model_name="Logistic (Balanced)",
    model_description="Logistic Regression Classifier",
    model_class=partial(
        LogisticRegression,
        max_iter=20000,
        penalty='l2',
        random_state=42,
        class_weight='balanced'
    )
)

register_model(
    model_name="LogisticCV",
    model_description="Logistic Regression Classifier with Cross Validation",
    model_class=partial(
        LogisticRegressionCV,
        max_iter=20000,
        penalty='l2',
        random_state=42,
        cv=5,
    )
)