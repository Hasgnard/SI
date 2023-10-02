
import numpy as np

def manhattan_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:

    return np.abs((x - y)).sum(axis=1)
