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
        # Extract patches from the input tensor using a sliding window
        patches = np.lib.stride_tricks.sliding_window_view(
            x_padded, (self.in_channels, kernel_height, kernel_width), axis=(1, 2, 3)
        )
        patches = patches[:, :, ::self.stride, ::self.stride, :, :, :].reshape(
            batch_size, out_height, out_width, -1
        )
        
        # Reshape weights for matrix multiplication
        weights_reshaped = self.weights.data.reshape(self.out_channels, -1).T
        
        # Perform matrix multiplication and add bias
        out = np.tensordot(patches, weights_reshaped, axes=([3], [0])) + self.bias.data
        out = out.transpose(0, 3, 1, 2)  # Rearrange dimensions to match output shape
        return Tensor(out, require_grad=True)
    
    def backward(self):
        self.weights.backward()
        self.bias.backward()
        