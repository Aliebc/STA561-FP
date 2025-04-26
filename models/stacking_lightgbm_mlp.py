from ._models import register_model
from functools import partial
from sklearn.ensemble import StackingClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

register_model(
    model_name="Stacking_LightGBM_MLP",
    model_description="Stacking Classifier (LightGBM + MLP -> LR)",
    model_class=partial(
        StackingClassifier,
        estimators=[
            ('lgbm', LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
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
