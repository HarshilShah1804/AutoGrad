from tensor import Tensor

class Relu():
    def __init__(self):
        pass

    def __repr__(self):
        return "ReLU()"
    
    def forward(self, x):
        if not isinstance(x, Tensor):
            try:
                x = Tensor(x)
            except Exception as e:
                raise TypeError(f"Input must be a Tensor, found {type(x)}, {e}")
        
        return x * (x > 0)  # Element-wise ReLU activation
    
    def backward(self, grad):
        if not isinstance(grad, Tensor):
            try:
                grad = Tensor(grad)
            except Exception as e:
                raise TypeError(f"Gradient must be a Tensor, found {type(grad)}, {e}")
        
        return grad * (self.data > 0)