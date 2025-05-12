import numpy as np
import math

class Tensor:
    def __init__(self, data, require_grad=False):
        self.data = np.array(data)
        self.require_grad = require_grad
        self.grad = None
        self.grad_fn = None
    
    def __repr__(self):
        return f"Tensor(data={self.data}, rqeuire_grad={self.require_grad})"


print(a)