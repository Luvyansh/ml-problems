import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        x = np.array(x)
        mean_x2 = np.mean(np.square(x))
        print('Mean of x^2 = ', mean_x2)
        rms = np.sqrt(mean_x2 + eps)
        print('RMS = ', rms)
        x_hat = x / rms
        return np.round(x_hat*gamma, 4)