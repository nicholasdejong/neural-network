class Cost:
    def __init__(self, forward, backward) -> None:
        self._forward = forward
        self._backward = backward

    def forward(self, expected, observed):
        return self._forward(expected, observed)

    def backward(self, expected, observed):
        return self._backward(expected, observed)

def default_forward(expected, observed):
    return 0.5 * (expected - observed) ** 2

def default_backward(expected, observed):
    return expected - observed

DefaultCost = Cost(default_forward, default_backward)
