#for forward and backward pass of layers

import numpy as np

class Layer:
    def __init__(self):
        self.parameters = {}
        self.gradients = {}

    def forward_pass(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Forward pass must be implemented.")
    
    def backward_pass(self, output_gradient: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Backward pass must be implemented.")