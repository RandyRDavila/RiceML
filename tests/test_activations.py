import numpy as np
import pytest
from rice_ml.activations import sign, sigmoid

def test_sign_positive():
    """Test sign function for positive input"""
    assert sign(3) == 1
    np.testing.assert_array_equal(sign([2, 5, 10]), np.array([1, 1, 1]))

def test_sign_negative():
    """Test sign function for negative input"""
    assert sign(-3) == -1
    np.testing.assert_array_equal(sign([-2, -5, -10]), np.array([-1, -1, -1]))

def test_sign_zero():
    """Test sign function for zero input"""
    assert sign(0) == 0
    np.testing.assert_array_equal(sign([0, 0]), np.array([0, 0]))

def test_sigmoid_array():
    """Test sigmoid function for array input"""
    result = sigmoid(np.array([0, 1, -1]))
    expected = np.array([0.5, 1 / (1 + np.exp(-1)), 1 / (1 + np.exp(1))])
    np.testing.assert_almost_equal(result, expected, decimal=5)
