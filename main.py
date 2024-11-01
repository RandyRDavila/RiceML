import seaborn as sns
import pandas as pd
import numpy as np

from rice_ml.classifiers import LogisticRegression
from rice_ml.load_data import iris_data


X, y = iris_data(
    features=["sepal_length", "petal_length"],
    species=["setosa", "virginica"],
    labels=[0, 1]
)

clf = LogisticRegression()

clf.train(X, y)

clf.plot_decision_boundary(X, y)
