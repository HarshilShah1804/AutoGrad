from tensor import Tensor
import numpy as np
from layers import Linear
from functions import Relu

class SGD:
    def __init__(self, parameters, lr=1e-3):
        self.parameters = parameters
        self.lr = lr
        self.state = {param: {'step': 0} for param in parameters}
        self.state['step'] = 0
        self.state['lr'] = lr
        self.state['momentum'] = 0.9
        self.state['weight_decay'] = 0.01
        self.state['nesterov'] = False

    def step(self):
        for param in self.parameters:
            if param.grad is None:
                continue

            # If param.grad shape doesn't match param.data, sum over batch dimension
            if param.grad.shape != param.data.shape:
                grad = param.grad.sum(axis=0)
            else:
                grad = param.grad

            # Apply weight decay
            if self.state['weight_decay'] != 0:
                grad += self.state['weight_decay'] * param.data

            # Update parameters
            if self.state['nesterov']:
                param.data -= self.state['lr'] * (grad + self.state['momentum'] * param.data)
            else:
                param.data -= self.state['lr'] * grad

            # Reset gradient
            param.grad = 0
            self.state['step'] += 1
            self.state[param]['step'] += 1
            self.state[param]['lr'] = self.state['lr']
            self.state[param]['momentum'] = self.state['momentum']
            self.state[param]['weight_decay'] = self.state['weight_decay']
            self.state[param]['nesterov'] = self.state['nesterov']