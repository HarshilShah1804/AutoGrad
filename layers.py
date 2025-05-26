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

class Conv2d:
    """
    Simple 2D Convolutional layer
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.weights = Tensor(np.random.randn(out_channels, in_channels, kernel_size, kernel_size), require_grad=True)
        self.bias = Tensor(np.random.randn(out_channels), require_grad=True)

    def __repr__(self, ):
        return (f"Conv2d(in_channels={self.in_channels}, out_channels={self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})")
    
    def forward(self, x):
        if not isinstance(x, Tensor):
            try:
                x = Tensor(x)
            except Exception as e:
                raise TypeError(f"Input must be a Tensor, found {type(x)}, {e}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Input shape {x.shape} does not match layer input shape {self.in_channels}")
        # Implementing a simple convolution operation
        batch_size, in_channels, height, width = x.shape
        kernel_height, kernel_width = self.kernel_size, self.kernel_size
        out_height = (height + 2 * self.padding - kernel_height) // self.stride + 1
        out_width = (width + 2 * self.padding - kernel_width) // self.stride + 1
        out = np.zeros((batch_size, self.out_channels, out_height, out_width))
        x_padded = np.pad(x.data, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for ic in range(self.in_channels):
                    for i in range(0, out_height):
                        for j in range(0, out_width):
                            h_start = i * self.stride
                            w_start = j * self.stride
                            out[b, oc, i, j] += np.sum(
                                x_padded[b, ic, h_start:h_start + kernel_height, w_start:w_start + kernel_width] *
                                self.weights.data[oc, ic]
                            )
                out[b, oc] += self.bias.data[oc]
        return Tensor(out, require_grad=True)
    
    def backward(self):
        self.weights.backward()
        self.bias.backward()
        