from typing import Callable, Union

import numpy as np 

from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.euclidean_distance import euclidean_distance
from si.metrics.rmse import rmse


class KNNRegressor:
    """
    K-Nearest Neighbors Regressor
    It is a supervised learning algorithm that can be used to solve both
    classification and regression problems.
    Estimates the average of the k most similar instances instead of 
    using the most frequent class.

    Parameters
    ----------
    k: int
        Number of neighbors to consider
    distance: Callable
        Distance function to use to compute the distance between two samples

    Attributes
    ----------
    dataset: np.ndarray
        Training data    
    
    """
    def __init__(self, k: int = 1, distance: Callable = euclidean_distance):
        '''
        Initializes the KNNRegressor

        Parameters
        ----------
        k: int
            Number of neighbors to consider
        distance: Callable
            Distance function to use to compute the distance between two samples
        '''

        # parameters
        self.k = k
        self.distance = distance

        # attributes
        self.dataset = None

    def fit(self, dataset: Dataset) -> 'KNNRegressor':
        '''
        Fits the KNNRegressor to the dataset

        Parameters
        ----------
        dataset: Dataset
            Dataset to fit the model to
        
        Returns
        -------
        self: KNNRegressor
            The fitted model
        '''

        self.dataset = dataset

        return self
    
    def _get_closest_label(self, sample: np.ndarray) -> Union[int, str]:
        '''
        Gets the label of the closest sample in the dataset to the given sample
        in the dataset.

        Parameters
        ----------
        sample: np.ndarray
            Sample to get the closest label to
        
        Returns
        -------
        label: Union[int, str]
            Label of the closest sample in the dataset.

        '''

        # get the distances between the sample and the training data
        distances = self.distance(sample, self.dataset.X)

        # get the k nearest neighbors
        k_nearest_neighbors = np.argsort(distances)[:self.k]

        # get the labels of the k nearest neighbors (in Y)
        k_nearest_neighbors_labels = self.dataset.y[k_nearest_neighbors]

        # get the average label
        label = np.mean(k_nearest_neighbors_labels)
        return label

    def predict(self, dataset: Dataset) -> np.ndarray:
        '''
        Predicts the labels of the given dataset

        Parameters
        ----------
        dataset: Dataset
            Dataset to predict the labels of

        Returns
        -------
        labels: np.ndarray
            Labels of the given dataset predicted by the model    
        '''

        # apply the _get_closest_label function to each sample in the dataset
        return np.apply_along_axis(self._get_closest_label, 1, dataset.X)
    
    def score(self, dataset: Dataset) -> float:
        '''
        Computes the accuracy of the model on the given dataset

        Parameters
        ----------
        dataset: Dataset
            Dataset to compute the accuracy on
        
        Returns
        -------
        accuracy: float
            RMSE of the model on the given dataset
        '''

        y_pred = self.predict(dataset)

        return rmse(dataset.y, y_pred)