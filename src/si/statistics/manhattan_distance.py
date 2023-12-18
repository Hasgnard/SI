
import numpy as np

def manhattan_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    '''
    calculates the Manhattan distance between X and Y using the following formula:
        distance_x_y1 =|x1 y11| + |x2 y12| + ... + | xn y1n|
        distance_x_y2 =|x1 y21| + |x2 y22| + ... + | xn y2n|
        ...
    
    Parameters
    ----------
    x : np.ndarray
        Single sample
    
    y : np.ndarray
        Multiple samples

    Returns
    -------
    np.ndarray
        Manhattan distance for each sample in Y
    '''

    return np.abs((x - y)).sum(axis=1)
