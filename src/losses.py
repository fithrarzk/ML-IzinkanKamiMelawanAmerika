import numpy as np

_EPS = 1e-15


class Loss:
    def forward(self, y_pred, y_true): raise NotImplementedError
    def backward(self, y_pred, y_true): raise NotImplementedError


class MSE(Loss):
    # L = (1/n) * Σ (ŷ - y)²
    def forward(self, y_pred, y_true):
        return float(np.mean((y_pred - y_true) ** 2))

    # dL/dŷ = (2/n) * (ŷ - y)
    def backward(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / y_pred.shape[0]


class BCE(Loss):
    # L = -(1/n) * Σ [ y·ln(ŷ) + (1-y)·ln(1-ŷ) ]
    def forward(self, y_pred, y_true):
        p = np.clip(y_pred, _EPS, 1 - _EPS)
        return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))

    # dL/dŷ = (1/n) * [ -y/ŷ + (1-y)/(1-ŷ) ]
    def backward(self, y_pred, y_true):
        p = np.clip(y_pred, _EPS, 1 - _EPS)
        return (-(y_true / p) + (1 - y_true) / (1 - p)) / y_pred.shape[0]


class CCE(Loss):
    # L = -(1/n) * Σ_i Σ_k y_ik · ln(ŷ_ik)  (one-hot targets, softmax output)
    def forward(self, y_pred, y_true):
        p = np.clip(y_pred, _EPS, 1.0)
        return float(-np.mean(np.sum(y_true * np.log(p), axis=1)))

    # Fused Softmax+CCE gradient: dL/dZ = (ŷ - y) / n
    def backward(self, y_pred, y_true):
        return (y_pred - y_true) / y_pred.shape[0]


LOSSES = {'mse': MSE, 'bce': BCE, 'cce': CCE}


def get_loss(name):
    if isinstance(name, Loss):
        return name
    try:
        return LOSSES[name.lower()]()
    except KeyError:
        raise ValueError(f"Unknown loss '{name}'. Choose from {list(LOSSES)}")
