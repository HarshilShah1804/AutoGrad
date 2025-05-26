import numpy as np
from tensor import Tensor

class Linear:
    """
    Simple feed-forward linear layer.
    """
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Tensor(np.random.randn(in_features, out_features), require_grad=True)
        self.bias = Tensor(np.random.randn(out_features), require_grad=True)

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features})"
    
    def forward(self, x):
        if not isinstance(x, Tensor):
            try:
                x = Tensor(x)
            except Exception as e:
                raise TypeError(f"Input must be a Tensor, found {type(x)}, {e}")
        
        if x.shape[1] != self.in_features:
            raise ValueError(f"Input shape {x.shape} does not match layer input shape {self.in_features}")
        
        return x @ self.weights + self.bias
    
    def backward(self):
        self.weights.backward()
        self.bias.backward()

