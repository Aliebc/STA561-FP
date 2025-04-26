from ._models import register_model
from functools import partial
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

register_model(
    model_name="Stacking_RF_LGBM_CatBoost_XGB_MLP",
    model_description="Stacking (RandomForest + LightGBM + CatBoost + XGBoost + MLP)",
    model_class=partial(
        StackingClassifier,
        estimators=[
            ('rf', RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42
            )),
            ('lgbm', LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                random_state=42
            )),
            ('cat', CatBoostClassifier(
                iterations=300,
                learning_rate=0.05,
                depth=6,
                verbose=0,
                random_seed=42
            )),
            ('xgb', XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                use_label_encoder=False,
                eval_metric='mlogloss',
                random_state=42
            )),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(100, 100),
                activation='relu',
                solver='adam',
                max_iter=300,
                random_state=42
            )),
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=3,
        n_jobs=-1,
        passthrough=True
    )
)
