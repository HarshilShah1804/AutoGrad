import numpy as np

class Tensor:
    def __init__(self, data, previous = None, require_grad=False):
        self.data = np.array(data)
        self.previous = previous
        self.require_grad = require_grad
        self.grad = None
        self.grad_fn = None
        self.shape = self.data.shape
    
    def __repr__(self):
        return f"Tensor(data={self.data}, require_grad={self.require_grad})"

    def __add__(self, other):
        if isinstance(other, Tensor):
            result = Tensor(self.data + other.data, previous=[self, other], require_grad=self.require_grad or other.require_grad)
        
            def grad_fn(grad):
                if self.reqiure_grad:
                    self.backward(grad)
                if other.require_grad:
                    other.backward(grad)
            result.grad_fn = grad_fn
            return result

    
    def __radd__(self, other):
        if isinstance(other, Tensor):
            result = Tensor(other.data + self.data, require_grad=self.require_grad or other.require_grad)
            return result
        else:
            result = Tensor(other + self.data, require_grad=self.require_grad)
            return result
        
    def __mul__(self, other):
        if isinstance(other, Tensor):
            result = Tensor(self.data * other.data, previous=[self, other], 
                            require_grad=self.require_grad or other.require_grad)
            def grad_fn(grad):
                if self.require_grad:
                    self.backward(grad * other.data)
                if other.require_grad:
                    other.backward(grad * self.data)
            return result
    
    def __rmul__(self, other):
        if isinstance(other, Tensor):
            result = Tensor(other.data * self.data, require_grad=self.require_grad or other.require_grad)
            return result
        else:
            result = Tensor(other * self.data, require_grad=self.require_grad)
            return result
        
    def __matmul__(self, other):
        if isinstance(other, Tensor):
            result = Tensor(self.data @ other.data, previous=[self, other], 
                            require_grad=self.require_grad or other.require_grad)
            def grad_fn(grad):
                if self.require_grad:
                    self.backward(grad @ other.data.T)
                if other.require_grad:
                    other.backward(self.data.T @ grad)
            result.grad_fn = grad_fn
            return result
        else:
            raise TypeError(f"Unsupported type for matmul: {type(other)}")

    def backward(self, grad=None):
        if self.require_grad == False:
            raise ValueError(f"require_grad is False, cannot call backward")
        if grad is None:
            grad = np.ones_like(self.data)
        self.grad += grad
        if self.grad_fn is not None:
            self.grad_fn(grad)
    
        