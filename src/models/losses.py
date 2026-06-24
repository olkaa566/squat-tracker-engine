import numpy as np

class BinaryCrossEntropy:
    def forward_pass(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1e-7) # (0,1)

        return np.mean( -(y_true * np.log(y_pred_clipped) + (1 - y_true) * (np.log(1 - y_pred_clipped))))
    
    def backward_pass(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1e-7) # (0,1)

        return ((y_pred_clipped - y_true) / (y_pred_clipped * (1 - y_pred_clipped))) / y_pred.shape[0] #calculating the avg error across a batch of images