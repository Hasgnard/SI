
import numpy as np
import sklearn.metrics as metrics



def rmse(y_true: np.array, y_pred: np.array) -> float:
    '''
    It computes the root mean squared error between two arrays
    
    Parameters
    ----------
    y_true: np.array
        The true values
        
    y_pred: np.array
        The predicted values
    
        
    Returns
    -------
    float
        The root mean squared error between the real and predicted values of y
    '''

    return np.sqrt(((y_true - y_pred)**2).mean())


if __name__ == '__main__':
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])

    y_true2 = np.array([0.5, 1,-1, 1,7, -6])
    y_pred2 = np.array([0, 2,-1, 2,8, -5])

    print(rmse(y_true2, y_pred2))
    print(metrics.mean_squared_error(y_true2, y_pred2))

