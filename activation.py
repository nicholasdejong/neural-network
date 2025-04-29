### Contains some Activation Functions for you to use.
### An implemented activation function needs a
### first derivative as well, for backpropagation purposes.
### Your activation function needs to be vectorized- in
### other words, it actually accepts a vector, not a
### single number. Don't worry, with numpy this is really
### easy. Take a look at some predefined examples to see
### what I mean.

import numpy as np

### Our Activation class used by Neural Network
class Activation:
    def __init__(self, forward, backward) -> None:
        self._forward = forward # f(x)
        self._backward = backward # f'(x)

    def forward(self, x):
        return self._forward(x)

    def backward(self, x):
        return self._backward(x)

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
