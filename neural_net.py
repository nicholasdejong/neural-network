import numpy as np
from activation import Tanh, Linear
from cost import DefaultCost

class NeuralNetwork():
    weights = []
    biases = []
    activations = [Linear]
    layers = []
    learning_rate = 0.005
    _cost = DefaultCost
    L = 0
    def __init__(self, layers) -> None:
        """ Initialize a new Neural Network with layer
            sizes corresponding to `layers`. """
        self.layers = layers
        self.L = len(layers) - 1 # Exclude input neurons

        # Initialize weights
        for i in range(self.L):
            self.weights += [np.random.random((layers[i], layers[i+1]))]

        # Initialize biases
        for layer in layers[1:]:
            self.biases += [np.random.random((1, layer))]

        # Initialize activations
        self.activations += [Linear] * self.L

    def activation(self, act, i):
        self.activations[i] = act

    def cost(self, test):
        """ Return the cost of the NN """
        sum = 0
        for [x, y] in test:
            xL = self.forward(x)
            sum += self._cost.forward(y, xL)
        sum /= len(test)
        return sum

    def forward(self, x):
        L = self.L
        xs = [x]
        acts = [x] # List of activated layers
        for i in range(1, L+1):
            x_next = np.matmul(xs[i-1], self.weights[i-1]) + self.biases[i-1]
            xs += [x_next]
            act_next = self.activations[i].forward(x_next)
            acts += [act_next]
        return xs[len(xs)-1]

    def train(self, batch, epocs):
        """ Train the Neural Network with data in `batch`. """
        for _ in range(epocs):
            L = self.L
            delta_weights = [
                np.zeros((self.layers[i], self.layers[i+1]))
                for i in range(L)
            ]
            delta_biases = [np.zeros((1,l)) for l in self.layers[1:]]
            for [x, y] in batch:
                # Forward propagation
                xs = [x]
                acts = [x] # List of activated layers
                for i in range(1, L+1):
                    x_next = np.matmul(xs[i-1], self.weights[i-1]) + self.biases[i-1]
                    xs += [x_next]
                    act_next = self.activations[i].forward(x_next)
                    acts += [act_next]
                # Backward propagation
                cost_wrt_z = DefaultCost.backward(y, acts[L])
                for i in range(L, 0, -1):
                    act = self.activations[i].backward(xs[i])
                    # print(act)
                    # act = act.reshape((1, np.shape(act)[0]))
                    cost_wrt_x = cost_wrt_z * act
                    cost_wrt_w = np.transpose(acts[i-1]) @ cost_wrt_x
                    cost_wrt_b = cost_wrt_x
                    delta_weights[i-1] += cost_wrt_w * self.learning_rate
                    delta_biases[i-1]  += cost_wrt_b * self.learning_rate
                    cost_wrt_z = (cost_wrt_z * act) @ np.transpose(self.weights[i-1])

            for i in range(len(delta_weights)):
                self.weights[i] += delta_weights[i]

            for i in range(len(delta_biases)):
                self.biases[i] += delta_biases[i]



nn = NeuralNetwork([2,3,2])
nn.activation(Tanh, 1)
nn.train([[np.array([[0,1]]), np.array([[1,2]])]], epochs=100)
print(nn.cost([[np.array([[0,1]]), np.array([[1,2]])]]))

