import numpy as np

__all__ = ['mse', 'binary_cross_entropy']

def mse(y_hat, y):
    """
    Computes the mean squared error between the predicted y_hat and the actual y.

    Parameters
    ----------
    y_hat : numpy.ndarray
        The predicted y values.

    y : numpy.ndarray
        The actual y values.

    Returns
    -------
    float
        The mean squared error.
    """
    return .5*((y_hat - y)**2).sum()

def binary_cross_entropy(y_hat, y):
    """
    Computes the binary cross entropy between the predicted y_hat and the actual y.

    Parameters
    ----------
    y_hat : numpy.ndarray
        The predicted y values.

    y : numpy.ndarray
        The actual y values.

    Returns
    -------
    float
        The binary cross entropy.
    """
    return -np.sum(y*np.log(y_hat) + (1 - y)*np.log(1 - y_hat))
