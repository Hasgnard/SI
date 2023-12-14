import numpy as np

from si.data.dataset import Dataset
from si.metrics.mse import mse


class RidgeRegressionLeastSquares:
    """
    The Ridge Regression Least Squares model employs the least 
    squares method to fit the model to the dataset. 
    It incorporates L2 regularization, which penalizes large 
    coefficients to mitigate overfitting and improve the model's 
    generalization performance.

    Parameters
    ----------
    l2_penalty: float
        The L2 regularization parameter
    scale: bool
        Whether to scale the dataset or not

    Attributes
    ----------
    theta: np.ndarray
        The model parameters
        For example, x0*
    theta_zero: float
        The intercept term
        for example, theta_zero 
    """

    def __init__(self, l2_penalty: float = 1, scale: bool = True):

        """
        Initializes the RidgeRegressionLeastSquares model.

        Parameters
        ----------
        l2_penalty: float
            The L2 regularization parameter
        scale: bool
            Whether to scale the dataset or not
        """
    
        # parameters
        self.l2_penalty = l2_penalty
        self.scale = scale

        # attributes
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None

    def fit(self, dataset: Dataset) -> 'RidgeRegressionLeastSquares':

        """
        It fits the model to the given dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: RidgeRegressionLeastSquares
            The fitted model
        """

        # scale the dataset

        if self.scale:
            # compute the mean and std
            self.mean = np.nanmean(dataset.X, axis=0)
            self.std = np.nanstd(dataset.X, axis=0)
            # scale the dataset
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        # Add intercept term to X
        X = np.c_[np.ones(X.shape[0]), X]

        # Compute the (penalty term l2_penalty * identity matrix)
        penalty = np.eye(X.shape[1]) * self.l2_penalty
        
        # Change the first position of the penalty matrix to 0
        penalty[0, 0] = 0

        # Compute the model parameters (theta_zero (first element of the theta vector) and theta (remaining elements))
        self.theta = np.linalg.inv(X.T.dot(X) + penalty).dot(X.T).dot(dataset.y)
        self.theta_zero = self.theta[0]
        self.theta = self.theta[1:]

        return self
    

    def predict(self, dataset: Dataset) -> np.ndarray:

        """
        It predicts the values of the given dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the values of

        Returns
        -------
        y_pred: np.ndarray
            The predicted values
        """
        # scale the dataset
        if self.scale:
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        # Add intercept term to X
        X = np.c_[np.ones(X.shape[0]), X]

        # Compute the predicted values
        y_pred = X.dot(np.r_[self.theta_zero, self.theta])

        return y_pred


    def score(self, dataset: Dataset) -> float:

        """
        Compute the mse score using the mse function
        
        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the score

        Returns
        -------
        score: float
            The score of the model
        """
        return mse(dataset.y, self.predict(dataset))

# This is how you can test it against sklearn
if __name__ == '__main__':
    
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    dataset_ = Dataset(X=X, y=y)

    # fit the model
    model = RidgeRegressionLeastSquares()
    model.fit(dataset_)
    print(model.theta)
    print(model.theta_zero)

    # compute the score
    print(model.score(dataset_))

    # compare with sklearn
    from sklearn.linear_model import Ridge
    model = Ridge()
    # scale data
    X = (dataset_.X - np.nanmean(dataset_.X, axis=0)) / np.nanstd(dataset_.X, axis=0)
    model.fit(X, dataset_.y)
    print(model.coef_) # should be the same as theta
    print(model.intercept_) # should be the same as theta_zero
    print(mse(dataset_.y, model.predict(X)))
