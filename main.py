from neural_net import NeuralNetwork
import numpy as np

nn = NeuralNetwork([2, 1])

nn.learning_rate(0.005)

def arr(x): return np.array([x])

dataset = []
N = 50
for _ in range(N):
    x = np.random.random((1, 2))
    y = np.array([[np.sum(x)]])
    dataset += [[x, y]]

test = [[arr([0.5, 0.5]), arr([1])]] # 0.5 + 0.5 == 1

cost = nn.cost(test)
print("Cost Before: ", cost)
nn.train(dataset, epochs=500)

cost = nn.cost(test)
print("Cost After: ", cost) # We want to try to minimize this.

# Print the NN's prediction for the first testcase.
print(nn.forward(test[0][0])) # Should be close to 1
