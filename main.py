from neural_net import NeuralNetwork
from activation import Tanh
import numpy as np

## Create a neural network with 3 layers:
# Input layer with 2 neurons
# Hidden layer with 3 neurons
# Output layer with 1 neuron
nn = NeuralNetwork([2,3,1])
# Tell the first layer (0-indexed) to use the Tanh activation function
nn.activation(Tanh, 1)

# Set the learning rate (defaults to 0.005)
nn.learning_rate(0.005)

# Define the data set to use
# Let's try the XOR Dataset!
def arr(x): return np.array([x])
dataset = [
    [arr([0, 0]), arr([0])],
    [arr([0, 1]), arr([1])],
    [arr([1, 0]), arr([1])],
    [arr([1, 1]), arr([0])]
]

# Train our Neural Network on our dataset for 100 epochs.
nn.train(dataset, epochs=100)

test = [[arr([0, 1]), arr([1])]]
cost = nn.cost(test)
print("Cost: ", cost) # We want to try to minimize this.

# Print the NN's prediction for the first testcase.
print(nn.forward(test[0][0])) # Should be close to 1
