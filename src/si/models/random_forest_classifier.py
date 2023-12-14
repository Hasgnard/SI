from typing import Literal
import numpy as np

from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy

class RandomForestClassifier:

    """
    The RandomForestClassifier model is an ensemble model that fits a number of decision 
    trees on various sub-samples of the dataset and uses averaging to improve the predictive
    accuracy and control over-fitting.
    """

    def __init__(self, n_estimators: int = 100, 
                 max_features: float = None, 
                 min_sample_split: int = 2, 
                 seed: int = 42, 
                 max_depth: int = 10, 
                 mode: Literal['gini', 'entropy'] = 'gini') -> None:

        """
        Initializes the RandomForestClassifier model.

        Parameters
        ----------
        n_estimators: int
            The number of trees in the forest.
        max_features: float
            The number of features to consider when looking for the best split
        min_sample_split: int
            The minimum number of samples required to split an internal node
        seed: int
            The seed to ensure reproducibility
        max_depth: int
            The maximum depth of the tree
        mode: str
            The mode to use to specify the criterion for splitting in the decision trees
        """

        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed

        # Estimated parameters
        self.tree =  []


    def fit(self, dataset: Dataset) -> 'RandomForestClassifier':
    
        ''' Fit the model to the training data
        
        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to
        
        Returns
        -------
        self: RandomForestClassifier
            The fitted model
        '''

        np.random.seed(self.seed)

        n_samples, n_features = dataset.shape()

        if self.max_features is None:
              self.max_features = int(np.sqrt(n_features))
        
        for i in range(self.n_estimators):
            bootstrap_samples = np.random.choice(n_samples, size=n_samples, replace=True)
            bootstrap_features = np.random.choice(n_features, size=self.max_features, replace=False)

            bootstrap_dataset = Dataset(X=dataset.X[bootstrap_samples][:, bootstrap_features], y=dataset.y[bootstrap_samples])

            tree = DecisionTreeClassifier(max_depth=self.max_depth, mode=self.mode, min_sample_split=self.min_sample_split)

            tree.fit(bootstrap_dataset)
        
            self.tree.append((tree, bootstrap_features))
        
        return self

    def predict(self, dataset: Dataset) -> np.ndarray:

        """
        Make predictions on the given dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to make predictions on
        
        Returns
        -------
        np.ndarray
            The predictions for the given dataset
        """
        ind_tree_predictions = []

        # get the predictions for each tree
        for tree, features in self.tree:
            tree_predictions = tree.predict(Dataset(X=dataset.X[:, features], y = dataset.y))
            ind_tree_predictions.append(tree_predictions)

        # get the most common prediction for each sample
        most_common_pred = [max(set(z), key=z.count) for z in zip(*ind_tree_predictions)]
        
        return np.array(most_common_pred)


    def score(self, dataset: Dataset) -> float:

        """
        Returns the accuracy of the model on the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to score the model on

        Returns
        -------
        float
            The accuracy of the model on the dataset

        """
    
        predictions = self.predict(dataset)

        return accuracy(dataset.y, predictions)



if __name__ == '__main__':
    from si.io.csv_file import read_csv
    from si.model_selection.split import train_test_split

    data = read_csv('C:\\Users\\Ruben Fernandes\\Documents\\GitHub\\GitHub\\si\\datasets\\iris\\iris.csv', sep=',', features=True, label=True)
    train, test = train_test_split(data, test_size=0.33, random_state=42)
    model = RandomForestClassifier(min_sample_split=3, max_depth=3, mode='gini', n_estimators=5)
    model.fit(train)
    print(model.score(test))


    from sklearn.ensemble import RandomForestClassifier as RFC
    
    model = RFC(n_estimators=5, max_depth=3, min_samples_split=3)
    model.fit(train.X, train.y)
    print(model.score(test.X, test.y))