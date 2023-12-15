from abc import abstractmethod
from typing import Union

import numpy as np
#import unittest

from si.neural_networks.layers import Layer


class ActivationLayer(Layer):
    """
    Base class for activation layers.
    """

    def forward_propagation(self, input: np.ndarray, training: bool) -> np.ndarray:
        """
        Perform forward propagation on the given input.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.
        training: bool
            Whether the layer is in training mode or in inference mode.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        self.input = input
        self.output = self.activation_function(self.input)
        return self.output

    def backward_propagation(self, output_error: float) -> Union[float, np.ndarray]:
        """
        Perform backward propagation on the given output error.

        Parameters
        ----------
        output_error: float
            The output error of the layer.

        Returns
        -------
        Union[float, numpy.ndarray]
            The output error of the layer.
        """
        return self.derivative(self.input) * output_error

    @abstractmethod
    def activation_function(self, input: np.ndarray) -> Union[float, np.ndarray]:
        """
        Activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        Union[float, numpy.ndarray]
            The output of the layer.
        """
        raise NotImplementedError

    @abstractmethod
    def derivative(self, input: np.ndarray) -> Union[float, np.ndarray]:
        """
        Derivative of the activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        Union[float, numpy.ndarray]
            The derivative of the activation function.
        """
        raise NotImplementedError

    def output_shape(self) -> tuple:
        """
        Returns the output shape of the layer.

        Returns
        -------
        tuple
            The output shape of the layer.
        """
        return self._input_shape

    def parameters(self) -> int:
        """
        Returns the number of parameters of the layer.

        Returns
        -------
        int
            The number of parameters of the layer.
        """
        return 0


class SigmoidActivation(ActivationLayer):
    """
    Sigmoid activation function.
    """

    def activation_function(self, input: np.ndarray):
        """
        Sigmoid activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        return 1 / (1 + np.exp(-input))

    def derivative(self, input: np.ndarray):
        """
        Derivative of the sigmoid activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The derivative of the activation function.
        """
        return self.activation_function(input) * (1 - self.activation_function(input))


class ReLUActivation(ActivationLayer):
    """
    ReLU activation function.
    """

    def activation_function(self, input: np.ndarray):
        """
        ReLU activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        return np.maximum(0, input)

    def derivative(self, input: np.ndarray):
        """
        Derivative of the ReLU activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The derivative of the activation function.
        """
        return np.where(input > 0, 1, 0)


class TanhActivation(ActivationLayer):
    """
    The tanh activation layer in NNs applies the hyperbolic tangent function to the output of
    neurons, squashing the values to the range of -1 to 1.
    """

    def activation_function(self, input: np.ndarray):
        """
        Tanh activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        #The tanh activation layer in NNs applies the hyperbolic tangent function to the output of neurons, squashing the values to the range of -1 to 1.
        return np.tanh(input)
        

    def derivative(self, input: np.ndarray):
        """
        Derivative of the tanh activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The derivative of the activation function.
        """
        return 1 - np.tanh(input) ** 2
    

class SoftmaxActivation(ActivationLayer):
    """
    The softmax activation layer in NNs transforms the raw output scores into a probability
    distribution (that sums to 1), making it suitable for multi-class classification problems.
    """

    def activation_function(self, input: np.ndarray):
        """
        Softmax activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        # The softmax activation layer in NNs transforms the raw output scores into a probability
        # distribution (that sums to 1), making it suitable for multi-class classification problems.
        
        exp_input = np.exp(input - np.max(input, keepdims=True))
        return exp_input / np.sum(exp_input, keepdims=True)
    

    def derivative(self, input: np.ndarray):
        """
        Derivative of the softmax activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The derivative of the activation function.
        """
        softmax = self.activation_function(input)
        return softmax * (1 - softmax)
    


# Test cases for the activation functions
# class TestActivationFunctions(unittest.TestCase):

#     def test_tanh_activation(self):
#         tanh_activation = TanhActivation()

#         # Test forward pass
#         input_values = np.array([1.0, 2.0, 3.0])
#         output_tanh = tanh_activation.activation_function(input_values)
#         expected_output_tanh = np.tanh(input_values)
#         np.testing.assert_array_almost_equal(output_tanh, expected_output_tanh)

#         # Test backward pass (derivative)
#         derivative_tanh = tanh_activation.derivative(input_values)
#         expected_derivative_tanh = 1 - np.tanh(input_values) ** 2
#         np.testing.assert_array_almost_equal(derivative_tanh, expected_derivative_tanh)

#     def test_softmax_activation(self):
#         softmax_activation = SoftmaxActivation()

#         # Test forward pass
#         input_values_softmax = np.array([1.0, 2.0, 3.0])
#         output_softmax = softmax_activation.activation_function(input_values_softmax)

#         # Ensure the output sums to 1
#         self.assertAlmostEqual(np.sum(output_softmax), 1.0)

#         # Test backward pass (derivative)
#         derivative_softmax = softmax_activation.derivative(input_values_softmax)

#         self.assertEqual(derivative_softmax.shape, input_values_softmax.shape)

if __name__ == '__main__':
    import tensorflow as tf

    # generate random input
    input_values = np.random.randn(1, 10)

    # test tanh activation
    tanh_activation = TanhActivation()
    output_tanh = tanh_activation.activation_function(input_values)
    expected_output_tanh = np.tanh(input_values)

    # test tensorflow tanh activation
    output_tanh_tf = tf.nn.tanh(input_values)
    
    print("SI Tanh activation forward pass output: ", output_tanh)
    print("Tensorflow tanh activation forward pass output: ", output_tanh_tf)
    print("Expected tanh activation forward pass output: ", expected_output_tanh)

    # test tanh activation derivative
    derivative_tanh = tanh_activation.derivative(input_values)
    print("SI Tanh activation derivative output: ", derivative_tanh)
    
    # test softmax activation
    softmax_activation = SoftmaxActivation()
    output_softmax = softmax_activation.activation_function(input_values)

    # test tensorflow softmax activation
    output_softmax_tf = tf.nn.softmax(input_values)

    print("SI Softmax activation forward pass output: ", output_softmax)
    print("Tensorflow softmax activation forward pass output: ", output_softmax_tf)

    # ensure the output sums to 1
    print("Sum of SI softmax activation output: ", np.sum(output_softmax))
    print("Sum of tensorflow softmax activation output: ", np.sum(output_softmax_tf))




