from typing import Literal, Tuple, Union, List

from si.models.decision_tree_classifier import DecisionTreeClassifier

class RandomForestClassifier:

    """

    """

    def __init__(self, n_estimators: int = 100, max_features: float = None, min_sample_split: int = 2, max_depth: int = 10, mode: Literal['gini', 'entropy'] = 'gini') -> None:

        """

        """

        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode

        # Estimated parameters
        self.tree = List[Tuple[DecisionTreeClassifier, List[]]] = []


        def fit(self, X: np.ndarray, y: np.ndarray) -> None:

            """

            """
        
        def predict(self, X: np.ndarray) -> np.ndarray:

            """

            """
            
        def score(self, X: np.ndarray, y: np.ndarray) -> float:

            """

            """