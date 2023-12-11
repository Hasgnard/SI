from abc import abstractmethod
import unittest
import numpy as np


class Optimizer:

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.

        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray
            The gradient of the loss function with respect to the weights.

        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        raise NotImplementedError


class SGD(Optimizer):

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        """
        Initialize the optimizer.

        Parameters
        ----------
        learning_rate: float
            The learning rate to use for updating the weights.
        momentum:
            The momentum to use for updating the weights.
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.retained_gradient = None

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.

        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray
            The gradient of the loss function with respect to the weights.

        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        if self.retained_gradient is None:
            self.retained_gradient = np.zeros(np.shape(w))
        self.retained_gradient = self.momentum * self.retained_gradient + (1 - self.momentum) * grad_loss_w
        return w - self.learning_rate * self.retained_gradient


class Adam(Optimizer):

    def __init__(self, learning_rate: float = 0.01, beta_1: float = 0.9, beta_2: float = 0.999, epsilon: float = 1e-8):
        """
        Initialize the optimizer.

        Parameters
        ----------
        learning_rate: float
            The learning rate to use for updating the weights.
        beta_1: float
            The beta_1 parameter of the Adam optimizer.
        beta_2: float
            The beta_2 parameter of the Adam optimizer.
        epsilon: float
            The epsilon parameter of the Adam optimizer.
        """
        super().__init__(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.

        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray
            The gradient of the loss function with respect to the weights.

        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        # verify if m and v are initialized , if not initialize them as matrices of zeros;
        if self.m is None:
            self.m = np.zeros(np.shape(w))
        if self.v is None:
            self.v = np.zeros(np.shape(w))
        # update the parameters            
        self.t += 1
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * grad_loss_w
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * np.power(grad_loss_w, 2)
        m_hat = self.m / (1 - np.power(self.beta_1, self.t))
        v_hat = self.v / (1 - np.power(self.beta_2, self.t))
        return w - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
    

class TestAdamOptimizer(unittest.TestCase):


    def setUp(self):
        self.learning_rate = 0.01
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 1e-8
        self.optimizer = Adam(learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.epsilon)

    def test_initialization(self):
        self.assertEqual(self.optimizer.learning_rate, self.learning_rate)
        self.assertEqual(self.optimizer.beta_1, self.beta_1)
        self.assertEqual(self.optimizer.beta_2, self.beta_2)
        self.assertEqual(self.optimizer.epsilon, self.epsilon)
        self.assertIsNone(self.optimizer.m)
        self.assertIsNone(self.optimizer.v)
        self.assertEqual(self.optimizer.t, 0)

    def test_update(self):
        adam = Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        w = np.array([1, 2, 3])
        grad_loss_w = np.array([1, 2, 3])

        # Perform one update
        updated_weights = adam.update(w, grad_loss_w)

        # Assertions on updated weights
        expected_m = 0.9 * np.array([0.0, 0.0, 0.0]) + 0.1 * np.array([1, 2, 3])
        expected_v = 0.999 * np.array([0.0, 0.0, 0.0]) + 0.001 * np.power(np.array([1, 2, 3]), 2)
        m_hat = expected_m / (1 - np.power(0.9, 1))
        v_hat = expected_v / (1 - np.power(0.999, 1))
        expected_result = w - 0.01 * m_hat / (np.sqrt(v_hat) + 1e-8)

        self.assertTrue(np.allclose(updated_weights, expected_result, atol=1e-7))

        # Assertions on adam.m and adam.v after the update
        self.assertTrue(np.allclose(adam.m, expected_m, atol=1e-7))
        self.assertTrue(np.allclose(adam.v, expected_v, atol=1e-7))

        # Assertions on adam.t after the update
        self.assertEqual(adam.t, 1)

        # Ensure that m and v are updated properly on subsequent calls
        adam.update(w, grad_loss_w)
        expected_m = 0.9 * expected_m + 0.1 * np.array([1, 2, 3])
        expected_v = 0.999 * expected_v + 0.001 * np.power(np.array([1, 2, 3]), 2)
        self.assertTrue(np.allclose(adam.m, expected_m, atol=1e-7))
        self.assertTrue(np.allclose(adam.v, expected_v, atol=1e-7))
        self.assertEqual(adam.t, 2)

    def test_initialization_of_m_and_v(self):
        adam = Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        w = np.array([1, 2, 3])
        grad_loss_w = np.array([1, 2, 3])

        # Ensure that m and v are initially None
        self.assertIsNone(adam.m)
        self.assertIsNone(adam.v)

        # Ensure that m and v are updated properly on subsequent calls
        adam.update(w, grad_loss_w)
        self.assertIsNotNone(adam.m)
        self.assertIsNotNone(adam.v)
        self.assertEqual(adam.t, 1)

if __name__ == '__main__':
    unittest.main()                 
