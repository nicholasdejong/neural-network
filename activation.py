### Contains some Activation Functions for you to use.
### An implemented activation function needs a
### first derivative as well, for backpropagation purposes.

import numpy as np

### Our Activation class used by Neural Network
class Activation:
    def __init__(self, forward, backward) -> None:
        self._forward = forward # f(x)
        self._backward = backward # f'(x)

    def forward(self, f):
        return self._forward(f)

    def backward(self, fp):
        return self._backward(fp)

### Example: Hyperbolic Tangent ###
def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return (1 / np.cosh(x) ** 2).reshape((1, np.shape(x)[1]))

Tanh = Activation(tanh, tanh_prime)
# Can now use Tanh.forward() and Tanh.backward()
###################################


### Example: Linear Activation (No activation)
def linear(x):
    return x

def linear_prime(x):
    return np.ones((1, np.shape(x)[1]))

Linear = Activation(linear, linear_prime)
##############################################

C = 0.1 # Leak factor

def leaky(x):
    return np.maximum(x, x * C)

def leaky_prime(x):
    return 0.5 * ((1 - C) * np.sign(x) + C + 1)

Leaky = Activation(leaky, leaky_prime)
##############################################
