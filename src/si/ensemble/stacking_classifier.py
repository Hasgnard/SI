
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
import numpy as np


class StackingClassifier:

    def __init__(self, models: list, final_model: object) -> None:

        '''
        
        '''
        self.models = models
        self.final_model = final_model
        self.meta_features = None

       

    def fit(self, dataset: Dataset) -> 'StackingClassifier':

        ''' Fit the model to the training data
        
        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to
        
        Returns
        -------
        self: StackingClassifier
            The fitted model
        '''
        #train the initial set of models
        for model in self.models:
            model.fit(dataset)
        
        predictions = []

        #get the predictions from the initial set of models
        for model in self.models:
            predictions.append(model.predict(dataset))

        meta_features = np.column_stack(predictions)

        self.final_model.fit(Dataset(dataset.X, meta_features))

        return self
        

    def predict(self, dataset: Dataset) -> np.ndarray:

        ''' Predict the labels of the dataset
        
        Parameters
        ----------
        dataset: Dataset
            The dataset to predict
        
        Returns
        -------
        np.ndarray
            The predicted labels
        '''

        # Get predictions from the initial set of models

        initial_predictions = []

        for model in self.models:
            initial_predictions.append(model.predict(dataset))
        
        # Get the final predictions using the final model and the predictions of the initial set of models

        meta_features = np.column_stack(initial_predictions)

        final_predictions = self.final_model.predict(Dataset(dataset.X, meta_features))

        return final_predictions
    
    def score(self, dataset: Dataset) -> float:

        ''' Score the model on the dataset
        
        Parameters
        ----------
        dataset: Dataset
            The dataset to score the model on
        
        Returns
        -------
        float
            The model score
        '''

        predictions = self.predict(dataset) 

        return accuracy(dataset.y, predictions)


from si.io.csv_file import read_csv
from si.model_selection.split import stratified_train_test_split
from si.models.knn_classifier import KNNClassifier
from si.models.logistic_regression import LogisticRegression
from si.models.decision_tree_classifier import DecisionTreeClassifier

data = read_csv('C:\\Users\\Ruben Fernandes\\Documents\\GitHub\\GitHub\\si\\datasets\\breast_bin\\breast-bin.csv', sep=",",features=True,label=True)
train, test = stratified_train_test_split(data, test_size=0.20, random_state=42)

#knnregressor
knn = KNNClassifier(k=5)

#logistic regression
lr=LogisticRegression(l2_penalty=0.1, alpha=0.1, max_iter=1000)

#decisiontreee
dt=DecisionTreeClassifier(min_sample_split=2, max_depth=10, mode='gini')

#final model
final_model=KNNClassifier(k=5)
modelos=[knn,lr,dt]
exercise=StackingClassifier(modelos,final_model)
exercise.fit(train)
print(exercise.score(test))

#sklearn
from sklearn.ensemble import StackingClassifier as StackingClassifier_sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

#knnregressor
knn = KNeighborsClassifier(n_neighbors=5)

#logistic regression
lr=LogisticRegression(penalty='l2', C=0.1, max_iter=1000)

#decisiontreee
dt=DecisionTreeClassifier(min_samples_split=2, max_depth=10, criterion='gini')

#final model
final_model=KNeighborsClassifier(n_neighbors=5)
models=[('knn',knn),('lr',lr),('dt',dt)]
exercise=StackingClassifier_sklearn(estimators=models,final_estimator=final_model)
exercise.fit(train.X, train.y)
print(accuracy(test.y, exercise.predict(test.X)))