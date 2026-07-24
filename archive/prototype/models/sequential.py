import numpy as np

class Sequential:
    def __init__(self, layers: list = None):
        if layers is not None:
            self.layers = layers
        else:
            self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward_pass(self, inputs: np.ndarray) -> np.ndarray:
        output = inputs

        for layer in self.layers:
            output = layer.forward_pass(output)

        return output
    
    def backward_pass(self, output_gradient: np.ndarray) -> np.ndarray:
        gradient = output_gradient

        for layer in reversed(self.layers):
            gradient = layer.backward_pass(gradient)

        return gradient