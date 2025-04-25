from sklearn.linear_model import (
    LogisticRegression,
    LogisticRegressionCV,
)
from sklearn.base import ClassifierMixin,BaseEstimator
from ._models import register_model
from functools import partial

#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

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

register_model(
    model_name="DecisionTree",
    model_description="Decision Tree Classifier",
    model_class=partial(
        DecisionTreeClassifier,
        random_state=42,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
    )
)