from rice_ml.activations import *
from rice_ml.loss_functions import *
from rice_ml.neuron import *
from rice_ml.utils import *

__all__ = ['LogisticRegression']


class LogisticRegression:
    def __init__(self):
        self.neuron = SingleNeuron(sigmoid, binary_cross_entropy)

    def train(self, X, y, alpha=0.1, epochs=100):
        self.neuron.train(X, y, alpha, epochs)
        self.errors_ = self.neuron.errors_

    def predict(self, X):
        return self.neuron.predict(X)

    def plot_decision_boundary(self, X, y):
        plot_decision_boundary(self.neuron, X, y)
