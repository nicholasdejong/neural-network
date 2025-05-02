from neural_net import NeuralNetwork
from activation import Leaky
import numpy as np

# An example

nn = NeuralNetwork([5,5,5,2])
nn.activation(Leaky, 1)
nn.activation(Leaky, 2)

nn.learning_rate(0.0005)

def arr(x): return np.array([x])

def int2bin(x):
    bits = []
    for _ in range(5):
        bits = [x & 1] + bits
        x >>= 1
    return bits

def parity(bits):
    ones = 0
    for bit in bits:
        if bit == 1:
            ones += 1
    return ones % 2

def majority(bits):
    maj = 0
    for bit in bits:
        if bit == 1:
            maj += 1
        else:
            maj -= 1
    if maj > 0: return 1
    return 0

dataset = []
for i in range(32):
    bits = int2bin(i)
    par, maj = parity(bits), majority(bits)
    bits = np.array([bits])
    out = np.array([[par, maj]])
    dataset += [[bits, out]]
test = [
    [
        np.array([[1,0,1,0,1]]),
        np.array([[1, 1]])
    ]
]

cost = nn.cost(test)
print("Cost Before: ", cost)
nn.train(dataset, epochs=500)

cost = nn.cost(test)
print("Cost After: ", cost)

print(nn.forward(test[0][0]))
