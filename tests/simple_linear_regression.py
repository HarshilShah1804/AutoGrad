import numpy as np
from tensor import Tensor
from layers import Linear
from optimizers import SGD
from loss import mse_loss

X = Tensor(np.random.randn(100, 1), require_grad=False)
y_true = Tensor(2 * X.data + 1, require_grad=False)

layer = Linear(1, 1)
optimizer = SGD([layer.weights, layer.bias], lr=0.1)

# Training loop
for epoch in range(100):
    y_pred = layer.forward(X)
    loss = mse_loss(y_pred, y_true)
    # print(type(loss))
    loss.backward()

    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.data}")

test_input = Tensor(np.array([[5.0]]))
test_output = layer.forward(test_input)
print("Prediction for input 5.0:", test_output.data)
