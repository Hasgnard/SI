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
      
        model_cv_scores = k_fold_cross_validation(model, dataset, scoring, cv)

        randomized_search_output["scores"].append(model_cv_scores)



        if np.mean(model_cv_scores) > randomized_search_output["best_score"]:
            randomized_search_output["best_score"] = np.mean(model_cv_scores)
            randomized_search_output["best_hyperparameters"] = hyperparameters
            

    
    return randomized_search_output