from typing import Callable, Union

import numpy as np 

from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.euclidean_distance import euclidean_distance
from si.metrics.rmse import rmse


class KNNClassifier:
    """

    """
    def __init__(self, k: int = 1, distance: Callable = euclidean_distance):

        # parameters
        self.k = k
        self.distance = distance

        # attributes
        self.dataset = None

    def fit(self, dataset: Dataset) -> 'KNNClassifier':

        self.dataset = dataset

        return self
    
    def _get_closest_label(self, sample: np.ndarray) -> Union[int, str]:

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


        # apply the _get_closest_label function to each sample in the dataset
        return np.apply_along_axis(self._get_closest_label, 1, dataset.X)
    
    def score(self, dataset: Dataset) -> float:


        y_pred = self.predict(dataset)

        return rmse(dataset.y, y_pred)