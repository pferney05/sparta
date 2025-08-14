import numpy as np
from typing import Optional, Literal, Callable

interpFunc = Callable[[np.ndarray[float],np.ndarray[float],np.ndarray[float]],np.ndarray[float]]
evalFunc = Callable[[np.ndarray[float], int], float]