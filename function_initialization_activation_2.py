# Import packages
import numpy as np
import seaborn as sns


def initialize_parameters(layers_dims):
    """
    Initialize parameters dictionary.

    Weight matrices will be initialized to random values from uniform normal
    distribution.
    bias vectors will be initialized to zeros.

    Arguments
    ---------
    layers_dims : list or array-like
        dimensions of each layer in the network.

    Returns
    -------
    parameters : dict
        weight matrix and the bias vector for each layer.
    """
    np.random.seed(1)
    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(
            layers_dims[l], layers_dims[l - 1]) * 0.01
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

        assert parameters["W" + str(l)].shape == (
            layers_dims[l], layers_dims[l - 1])
        assert parameters["b" + str(l)].shape == (layers_dims[l], 1)

    return parameters

#--------------------------------------------------------------------------------

    # COMO IMPORTAMOS LA CLASE Activation_Function que se encuentra en el archivo function_initializatiun_activation_2.py
    # from function_initialization_activation_2 import Activation_Function as af
    # importo funcion
    # from function_cost_gradient import compute_cost 
    # importo funcion de una clase
    # from function_initialization_activation_2 import Activation_Function 


#--------------------------------------------------------------------------------

class Activation_Function:

    def __init__(self, Z):
        self.Z = Z

    # Define activation functions that will be used in forward propagation
    
    def sigmoid(self):
        """
        Computes the sigmoid of Z element-wise.

        Arguments
        ---------
        Z : array
            output of affine transformation.

        Returns
        -------
        A : array
            post activation output.
        Z : array
            output of affine transformation.
        """
        A = 1 / (1 + np.exp(-self.Z))

        return A, self.Z


    def tanh(self):
        """
        Computes the Hyperbolic Tagent of Z elemnet-wise.

        Arguments
        ---------
        Z : array
            output of affine transformation.

        Returns
        -------
        A : array
            post activation output.
        Z : array
            output of affine transformation.
        """
        A = np.tanh(self.Z)

        return A, self.Z


    def relu(self):
        """
        Computes the Rectified Linear Unit (ReLU) element-wise.

        Arguments
        ---------
        Z : array
            output of affine transformation.

        Returns
        -------
        A : array
            post activation output.
        Z : array
            output of affine transformation.
        """
        A = np.maximum(0, self.Z)

        return A, self.Z


    def leaky_relu(self):
        """
        Computes Leaky Rectified Linear Unit element-wise.

        Arguments
        ---------
        Z : array
            output of affine transformation.

        Returns
        -------
        A : array
            post activation output.
        Z : array
            output of affine transformation.
        """
        A = np.maximum(0.1 * self.Z, self.Z)

        return A, self.Z
