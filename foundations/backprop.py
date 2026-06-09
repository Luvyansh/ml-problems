import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        z = (x @ w) + b
        print('z = ', z)
        y_hat = np.round((1 / (1+np.exp(-z))), 5)
        print('y_hat = ', y_hat)
        error = np.round((y_hat - y_true), 5)
        print('error = ', error)
        dSig_dw = np.round((y_hat * (1.0 - y_hat)), 5)
        print('dSig_dw = ', dSig_dw)
        delta = np.round((error * dSig_dw), 5)
        print('delta = ', delta)
        dL_dw = np.round((delta * x), 5)
        dL_db = round((float(delta)), 5)
        return (dL_dw, dL_db)