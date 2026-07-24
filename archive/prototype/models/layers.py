"""
forward pass = inputs(data) * weights + bias: Z = XW + b
backward pass - using Chain Rule: 
    calculate the partial derivatives:
        dl / dW = X^T * dZ
        dl / db = sum(dZ)
    
    return gradient with respect to the input:
        dl / dX = dZ * W^T

"""

import numpy as np
from .base import Layer

class Dense(Layer):
    def __init__(self, input_size: int, output_size: int):
        super().__init__() #base constructor

        #initialize weights randomly and biases to zero
        self.parameters["W"] = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.parameters["b"] = np.zeros((1, output_size))

    def forward_pass(self, inputs: np.ndarray) ->np.ndarray:
        self.inputs = inputs

        return np.dot(inputs, self.parameters["W"]) + self.parameters["b"] #dot - matrix multiplication
    
    def backward_pass(self, output_gradient: np.ndarray) -> np.ndarray:
        self.gradients["W"] = np.dot(self.inputs.T, output_gradient) #T - transpose
        self.gradients["b"] = np.sum(output_gradient, axis = 0, keepdims = True)

        return np.dot(output_gradient, self.parameters["W"].T)
        
