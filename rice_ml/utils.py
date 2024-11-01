import numpy as np
import matplotlib.pyplot as plt

__all__ = ['plot_decision_boundary']

def plot_decision_boundary(neuron, X, y, resolution=0.01):
    """
    Plots the decision boundary for a 2D dataset using a trained neuron.

    Parameters
    ----------
    neuron : object
        A trained neuron object with a `predict` method that takes a 2D numpy array
        `X` as input and returns predictions. The neuron should have learned weights
        for a 2D input.

    X : numpy.ndarray
        A 2D array of shape (n_samples, 2) representing the input feature data.

    y : numpy.ndarray
        A 1D array of shape (n_samples,) representing the class labels for each sample in `X`.

    resolution : float, optional (default=0.01)
        The step size of the mesh grid. Smaller values result in finer boundaries
        but require more computation.

    Returns
    -------
    None
        Displays the decision boundary plot.
    """
    # Define the mesh grid boundaries based on X's min and max values
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Generate a mesh grid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))

    # Use the neuron's predict method to classify each point in the mesh grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = neuron.predict(grid_points)
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")

    # Scatter plot the actual data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, s=40, edgecolor="k", cmap="coolwarm", marker='o')
    legend1 = plt.legend(*scatter.legend_elements(), title="Classes")
    plt.gca().add_artist(legend1)

    # Labels and title
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision Boundary")
    plt.show()
