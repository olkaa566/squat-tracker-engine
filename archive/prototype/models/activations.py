import numpy as np
from .base import Layer

#rectified linear unit activation layer - return input(positive) or 0
class ReLU(Layer):
    def forward_pass(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs

        return np.maximum(0, inputs)
    
    def backward_pass(self, output_gradient: np.ndarray) -> np.ndarray:
        #derivative is 1 for positive, 0 for negative
        relu_gradient = self.inputs > 0

        return output_gradient * relu_gradient
    
#layer to give percentages between 0.0 and 1.0 (pass) at the end
class Sigmoid(Layer):
    def forward_pass(self, inputs: np.ndarray) -> np.ndarray:
        inputs = np.clip(inputs, -500, 500)
        self.output = 1 / (1 + np.exp(-inputs))

        return self.output
    
    def backward_pass(self, output_gradient: np.ndarray) -> np.ndarray:
        sigmoid_gradient = self.output * (1 - self.output)

        return sigmoid_gradient * output_gradient