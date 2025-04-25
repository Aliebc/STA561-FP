from sklearn.linear_model import (
    LogisticRegression,
    LogisticRegressionCV,
)
from sklearn.base import ClassifierMixin,BaseEstimator
from ._models import register_model

register_model(
    model_name="Logistic",
    model_description="Logistic Regression Classifier",
    model_class=LogisticRegression
)

register_model(
    model_name="LogisticCV",
    model_description="Logistic Regression Classifier with Cross Validation",
    model_class=LogisticRegressionCV
)