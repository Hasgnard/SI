import numpy as np


from si.data.dataset import Dataset

class PCA:
    '''
    Performs Principal Component Analysis (PCA) on a dataset to simplify 
    and reduce its dimensionality while retaining the essential information.

    Parameters
    ----------
    n_components: int, optional
        Number of principal components to retain (default is 2).

    Attributes
    ----------
    n_components: int
        Number of principal components.
    mean: np.ndarray
        Mean of each feature.
    components: np.ndarray
        Principal components.
    explained_variance: np.ndarray
        Explained variance of each principal component.
    '''
    def __init__(self, n_components: int = 2):
        

        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None


    def _get_centered_data(self, dataset: Dataset) -> np.ndarray:

        '''Center the data by subtracting the mean from each feature.
        
        Parameters
        ----------
        dataset: Dataset
            Dataset to be centered.
        
        Returns
        -------
        np.ndarray
            Centered data.'''

        self.mean = np.mean(dataset.X, axis=0)
        self.centered_data = dataset.X - self.mean

        return self.centered_data
    

    def _get_components(self, dataset: Dataset) -> np.ndarray:
        '''
        Get the principal components from the centered dataset.
        
        Returns
        -------
        np.ndarray
            Components.
        '''

        centered_data = self._get_centered_data(dataset)
        self.u_matrix, self.s_matrix, self.v_matrix_t = np.linalg.svd(centered_data, full_matrices=False)
        self.components = self.v_matrix_t[: , :self.n_components]

        return self.components
    
    
    def _get_explained_variance(self, dataset: Dataset) -> np.ndarray:
        '''
        Get the explained variance from the centered dataset.
        
        Returns
        -------
        np.ndarray
            Explained variance.
        '''

        ev_formula = self.s_matrix ** 2 / (len(dataset.X) - 1)
        explained_variance = ev_formula[:self.n_components]

        return explained_variance
    

    def fit(self, dataset: Dataset) -> 'PCA':
        '''
        Fit the PCA model to the dataset.

        Parameters
        ----------
        dataset: Dataset
            Dataset to be fitted.
        
        Returns
        -------
        PCA
            Fitted PCA model.
        ''' 

        self.components = self._get_components(dataset)
        self.explained_variance = self._get_explained_variance(dataset)

        return self
    

    def transform(self, dataset: Dataset) -> Dataset:
        '''
        Transform the dataset using the fitted PCA model.

        Parameters
        ----------
        dataset: Dataset
            Dataset to be transformed.

        Returns
        -------
        Dataset
            Transformed dataset.
        '''

        if self.components is None:
            raise Exception('PCA not fitted yet')
        
        v_matrix = self.v_matrix_t.T
        transformed_data = np.dot(self.centered_data, v_matrix)

        return Dataset(transformed_data, dataset.y, dataset.features, dataset.label)
    
    
    def fit_transform(self, dataset: Dataset) -> Dataset:
        '''
        Fit the PCA model to the dataset and transform it.

        Parameters
        ----------
        dataset: Dataset
            Dataset to be fitted and transformed.
        
        Returns
        -------
        Dataset
            Transformed dataset.
        '''
        
        self.fit(dataset)
        return self.transform(dataset)

