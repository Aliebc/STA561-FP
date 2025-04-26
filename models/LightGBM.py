from ._models import register_model
from functools import partial
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV

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
        force_row_wise=True,
        verbosity=-1
    )
)

class LightGBMWithGridSearch:
    def __init__(self, random_state=42):
        base_model = LGBMClassifier(
            random_state=random_state,
            force_row_wise=True,
            verbosity=-1
        )

        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, -1],
            'num_leaves': [15, 31, 63],
        }

        self.model = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring='accuracy',  # 或者你换成 'roc_auc', 'f1', 看你的需求
            cv=3,                # 3折交叉验证
            n_jobs=-1,           # 全部CPU核心
            verbose=1
        )

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def best_params_(self):
        if hasattr(self.model, 'best_params_'):
            return self.model.best_params_
        else:
            return None

# 注册新模型
register_model(
    model_name="LightGBM-AutoTuned",
    model_description="LightGBM Classifier with GridSearchCV hyperparameter tuning",
    model_class=LightGBMWithGridSearch
)