# AutoGrad

A simple educational framework for building and training neural networks from scratch using NumPy, with automatic differentiation support.

## Features

- Custom `Tensor` class with autograd support
- Basic neural network layers (e.g., Linear)
- Optimizers (e.g., SGD)
- Loss functions (e.g., MSE)
- Simple training and testing scripts
- Minimal dependencies (just NumPy)

## Getting Started

1. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2. **Run a test or example:**
    ```bash
    python tests/simple_linear_regression.py
    ```

3. **Run all tests:**
    ```bash
    python run_tests.py
    ```

## Continuous Integration

- All pushes and pull requests to `main` run the tests automatically via GitHub Actions.
- The workflow is defined in `.github/workflows/CI.yml`.

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.
