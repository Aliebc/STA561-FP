from ._models import register_model
from functools import partial
from sklearn.neural_network import MLPClassifier

register_model(
    model_name="MLP",
    model_description="Multi-layer Perceptron Classifier (1 hidden layer)",
    model_class=partial(
        MLPClassifier,
        hidden_layer_sizes=(100,),  
        activation='relu',
        solver='adam',
        random_state=42,
        max_iter=300
    )
)
