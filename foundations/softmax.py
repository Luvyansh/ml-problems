import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        n = z - np.max(z)
        d = np.exp(n)
        return np.round(d/(np.sum(d)), 4)