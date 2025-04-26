from ._models import register_model
from functools import partial
from sklearn.dummy import DummyClassifier

register_model(
    model_name="Dummy",
    model_description="Dummy Classifier",
    model_class=partial(
        DummyClassifier,
        strategy="most_frequent"
    )
)