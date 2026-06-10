import numpy as np
from typing import List


class Solution:
    def ReLU(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.maximum(x, 0)
    
    def MSE(self, y_hat: NDArray[np.float64], y_true: NDArray[np.float64]):
        return (y_hat - y_true) ** 2

    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        x = np.array(x, dtype=np.float64).reshape(1, -1)
        y_true = np.array(y_true, dtype=np.float64).reshape(1, -1)
        b1 = np.array(b1, dtype=np.float64).reshape(1, -1)
        b2 = np.array(b2, dtype=np.float64).reshape(1, -1)
        W1 = np.array(W1, dtype=np.float64).T
        W2 = np.array(W2, dtype=np.float64).T

        z1 = x @ W1 + b1
        a1 = self.ReLU(z1)
        y_hat = a1 @ W2 + b2

        loss = np.round(self.MSE(y_hat, y_true), 4)
        dL_dy_hat = 2 * (y_hat - y_true) / y_hat.shape[0]
        dL_dW2 = np.round(dL_dy_hat.T @ a1, 4)
        dL_db2 = np.round(dL_dy_hat.flatten(), 4)
        dL_da1 = dL_dy_hat @ W2.T
        dL_dz1 = dL_da1 * (z1 > 0)
        dL_dW1 = np.round(dL_dz1.T @ x, 4)
        dL_db1 = np.round(dL_dz1.flatten(), 4)

        return {
            "loss": loss[0][0],
            "dW1": dL_dW1.tolist(),
            "db1": dL_db1.tolist(),
            "dW2": dL_dW2.tolist(),
            "db2": dL_db2.tolist()
        }