import numpy as np


class Activation:
    def forward(self, Z): raise NotImplementedError
    def backward(self, Z): raise NotImplementedError


class Linear(Activation):
    def forward(self, Z):  return Z
    def backward(self, Z): return np.ones_like(Z)


class ReLU(Activation):
    def forward(self, Z):  return np.maximum(0, Z)
    def backward(self, Z): return (Z > 0).astype(float)


class Sigmoid(Activation):
    def forward(self, Z):
        return 1 / (1 + np.exp(-np.clip(Z, -500, 500)))
    def backward(self, Z):
        s = self.forward(Z)
        return s * (1 - s)


class Tanh(Activation):
    def forward(self, Z):  return np.tanh(Z)
    def backward(self, Z): return 1 - np.tanh(Z) ** 2


class Softmax(Activation):
    def forward(self, Z):
        e = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def backward(self, Z):
        S = self.forward(Z)
        batch, n = S.shape
        J = -np.einsum('bi,bj->bij', S, S)
        J[:, np.arange(n), np.arange(n)] += S
        return J   # (batch, n, n)


ACTIVATIONS = {
    'linear':  Linear,
    'relu':    ReLU,
    'sigmoid': Sigmoid,
    'tanh':    Tanh,
    'softmax': Softmax,
}


def get_activation(name):
    if isinstance(name, Activation):
        return name
    try:
        return ACTIVATIONS[name.lower()]()
    except KeyError:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(ACTIVATIONS)}")
