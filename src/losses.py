import numpy as np

_EPS = 1e-15


class Loss:
    def forward(self, y_pred, y_true): raise NotImplementedError
    def backward(self, y_pred, y_true): raise NotImplementedError


class MSE(Loss):
    def forward(self, y_pred, y_true):
        return float(np.mean((y_pred - y_true) ** 2))

    def backward(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / y_pred.shape[0]


class BCE(Loss):
    def forward(self, y_pred, y_true):
        p = np.clip(y_pred, _EPS, 1 - _EPS)
        return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))

    def backward(self, y_pred, y_true):
        p = np.clip(y_pred, _EPS, 1 - _EPS)
        return (-(y_true / p) + (1 - y_true) / (1 - p)) / y_pred.shape[0]


class CCE(Loss):
    def forward(self, y_pred, y_true):
        p = np.clip(y_pred, _EPS, 1.0)
        return float(-np.mean(np.sum(y_true * np.log(p), axis=1)))

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
