"""
Created on Fri Apr 25 12:21:39 2025

@author: wangb
"""

from ._models import register_model
from functools import partial
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

register_model(
    model_name="XGBoost_Tuned",
    model_description="XGBoost Classifier with GridSearchCV",
    model_class=partial(
        GridSearchCV,
        estimator=XGBClassifier(eval_metric='logloss'),
        param_grid={
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0]
        },
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
)
