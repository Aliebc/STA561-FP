from ._models import register_model
from functools import partial
from sklearn.tree import DecisionTreeClassifier

register_model(
    model_name="DecisionTree",
    model_description="Decision Tree Classifier",
    model_class=partial(
        DecisionTreeClassifier,
        random_state=42,
        max_depth=10,
    )
)