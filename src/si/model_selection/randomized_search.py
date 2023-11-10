from typing import Dict, Tuple, Callable
from si.data.dataset import Dataset
from si.model_selection.cross_validation import k_fold_cross_validation

import numpy as np



def randomized_search_cv(model,
                        dataset: Dataset,
                        hyperparameter_grid: Dict[str, Tuple],
                        scoring: Callable = None,
                        cv: int = 5,
                        n_iter: int = 10) -> dict:
    """
    Performs a randomized search cross validation on a model.

    Parameters
    ----------
    model
        The model to cross validate.
    dataset: Dataset
        The dataset to cross validate on.
    hyperparameter_grid: Dict[str, Tuple]
        The hyperparameter grid to use.
    scoring: Callable
        The scoring function to use.
    cv: int
        The cross validation folds.
    n_iter: int
        The number of iterations to perform.

    Returns
    -------
    results: Dict[str, Any]
        The results of the randomized search cross validation. Includes the scores, hyperparameters,
        best hyperparameters and best score.
    """
    # check if hyperparameter grid is valid
    
    for parameter in hyperparameter_grid:
        if not hasattr(model, parameter):
            raise AttributeError(f"Model {model} does not have parameter {parameter}.")
    
    

    randomized_search_output = {"hyperparameters": [],
                                "scores": [],
                                "best_hyperparameters": None,
                                "best_score": 0}
    
    # Get n_iter hyperparameter combinations
    
    for i in range(n_iter):
        hyperparameters = {}
        for parameter in hyperparameter_grid:
            hyperparameters[parameter] = np.random.choice(hyperparameter_grid[parameter])
        
        randomized_search_output["hyperparameters"].append(hyperparameters)
        
        for key, value in hyperparameters.items():
            setattr(model, key, value)
      
        model_cv_scores = k_fold_cross_validation(model, dataset, cv, scoring)

        randomized_search_output["scores"].append(model_cv_scores)



        if np.mean(model_cv_scores) > randomized_search_output["best_score"]:
            randomized_search_output["best_score"] = np.mean(model_cv_scores)
            randomized_search_output["best_hyperparameters"] = hyperparameters
            

    
    return randomized_search_output

    

if __name__ == '__main__':
    from si.models.logistic_regression import LogisticRegression
    from si.model_selection.grid_search import grid_search_cv
    from si.io.csv_file import read_csv


    # load the dataset
    dataset = read_csv('C:\\Users\\Ruben Fernandes\\Documents\\GitHub\\GitHub\\si\\datasets\\breast_bin\\breast-bin.csv', sep=",",features=True,label=True)

    # define the model
    model = LogisticRegression()

    # define the hyperparameter grid
    hyperparameter_grid = {'l2_penalty': np.linspace(1, 10, 10),
                           'alpha': np.linspace(0.001, 0.0001, 100),
                           'max_iter': np.linspace(1000, 2000, 200),
                           }
    # print(hyperparameter_grid)

    # perform grid search cross validation
    results = randomized_search_cv(model=model, dataset=dataset, hyperparameter_grid=hyperparameter_grid, cv=5, n_iter=10)
    
    # print the results
    print('Grid search results:\n')

    print(f'Best score:\n {results["best_scores"]}')
    print()
    print(f'Best hyperparameters:\n {results["best_hyperparameters"]}')
    print()
    print(f'All scores:\n {results["scores"]}')
    print()
    print(f'All hyperparameters:\n {results["hyperparameters"]}')