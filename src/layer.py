import numpy as np
from activations import get_activation


class DenseLayer:
    def __init__(self, n_in, n_out, activation='linear',
                 init_method='random_normal', init_params=None,
                 is_softmax_output=False):
        self.n_in  = n_in
        self.n_out = n_out
        self.activation       = get_activation(activation)
        self.is_softmax_output = is_softmax_output

        self.W  = np.zeros((n_in, n_out))
        self.b  = np.zeros((1, n_out))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self._X = None   # input cache
        self._Z = None   # pre-activation cache

        self._init_weights(init_method, **(init_params or {}))

    def _init_weights(self, method, **kw):
        method = method.lower()
        if method == 'zero':
            self.W = np.zeros((self.n_in, self.n_out))
            self.b = np.zeros((1, self.n_out))

        elif method == 'random_uniform':
            rng = np.random.default_rng(kw.get('seed'))
            self.W = rng.uniform(kw.get('low', 0), kw.get('high', 1),
                                 (self.n_in, self.n_out))
            self.b = rng.uniform(kw.get('low', 0), kw.get('high', 1),
                                 (1, self.n_out))

        elif method == 'random_normal':
            rng = np.random.default_rng(kw.get('seed'))
            std = np.sqrt(kw.get('var', 1.0))
            self.W = rng.normal(kw.get('mean', 0), std, (self.n_in, self.n_out))
            self.b = rng.normal(kw.get('mean', 0), std, (1, self.n_out))

        else:
            raise ValueError(f"Unknown init method '{method}'.")

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, X):
        self._X = X
        self._Z = X @ self.W + self.b
        return self.activation.forward(self._Z)

    def backward(self, dA):
        batch = self._X.shape[0]

        # For Softmax+CCE the fused gradient is passed directly as dZ
        dZ = dA if self.is_softmax_output else dA * self.activation.backward(self._Z)

        self.dW = self._X.T @ dZ / batch
        self.db = np.mean(dZ, axis=0, keepdims=True)
        return dZ @ self.W.T

    def update(self, lr, regularization='none', lambda_=0.0):
        reg = regularization.lower()
        if   reg == 'l2': reg_term = lambda_ * self.W
        elif reg == 'l1': reg_term = lambda_ * np.sign(self.W)
        else:             reg_term = 0.0

        self.W -= lr * (self.dW + reg_term)
        self.b -= lr * self.db

    def get_params(self):    return {'W': self.W, 'b': self.b}
    def get_gradients(self): return {'dW': self.dW, 'db': self.db}

    def set_params(self, W, b):
        self.W, self.b = W.copy(), b.copy()

    def __repr__(self):
        return (f"DenseLayer({self.n_in} → {self.n_out}, "
                f"activation={self.activation.__class__.__name__})")
