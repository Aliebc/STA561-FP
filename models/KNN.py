from ._models import register_model
from functools import partial
from sklearn.neighbors import KNeighborsClassifier

register_model(
    model_name="KNN",
    model_description="K-Nearest Neighbors Classifier (k=5)",
    model_class=partial(
        KNeighborsClassifier,
        n_neighbors=5
    )
)
