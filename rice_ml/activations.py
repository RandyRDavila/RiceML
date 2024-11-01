import numpy as np

__all__ = [
    'sign',
    'sigmoid',
    'constant',
]

def sign(x):
    """
    Computes the sign of each element in an array or scalar value.

    Parameters
    ----------
    x : array-like or scalar
        Input value(s) for which the sign is computed. This can be a scalar, a
        list, or a numpy array.

    Returns
    -------
    numpy.ndarray or scalar
        The sign of each element in the input `x`, where:
        - `1` indicates a positive number,
        - `-1` indicates a negative number, and
        - `0` indicates zero.

    Notes
    -----
    This function uses numpy's `np.sign`, which applies element-wise and is
    vectorized, making it efficient for arrays.

    Examples
    --------
    >>> sign(-5)
    -1
    >>> sign(0)
    0
    >>> sign([3, -2, 0, 5])
    array([ 1, -1,  0,  1])
    """
    return np.sign(x)

def sigmoid(x):
    """
    Applies the sigmoid activation function to each element in an array or scalar value.

    The sigmoid function maps any real-valued number into the range (0, 1),
    making it useful in binary classification models and neural networks.

    Parameters
    ----------
    x : array-like or scalar
        Input value(s) for which the sigmoid is computed. Can be a scalar, list,
        or numpy array.

    Returns
    -------
    numpy.ndarray or scalar
        The sigmoid of each element in the input `x`, where each value is
        mapped between 0 and 1.

    Notes
    -----
    The sigmoid function is defined as:
        sigmoid(x) = 1 / (1 + exp(-x))
    It is often used as an activation function in neural networks.

    Examples
    --------
    >>> sigmoid(0)
    0.5
    >>> sigmoid([1, -1, 0])
    array([0.73105858, 0.26894142, 0.5       ])
    """
    return 1 / (1 + np.exp(-x))

def constant(x):
    """
    Returns the input value(s) without modification.

    This function can be used as a placeholder or to apply a "constant"
    activation function where the output is the same as the input.

    Parameters
    ----------
    x : any
        Input value(s) of any data type. Commonly, this is a numeric scalar, list,
        or numpy array.

    Returns
    -------
    same as input
        The input `x` is returned without any modification.

    Examples
    --------
    >>> constant(5)
    5
    >>> constant([1, 2, 3])
    [1, 2, 3]
    """
    return x
