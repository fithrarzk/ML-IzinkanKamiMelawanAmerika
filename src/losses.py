import numpy as np
from autograd import Tensor

_EPS = 1e-15

class Loss:
    def forward(self, y_pred: Tensor, y_true: np.ndarray) -> Tensor:
        raise NotImplementedError

    def __call__(self, y_pred: Tensor, y_true: np.ndarray) -> Tensor:
        return self.forward(y_pred, y_true)

class MSE(Loss):
    def forward(self, y_pred: Tensor, y_true: np.ndarray) -> Tensor:
        # Standard MSE aja: mean((ŷ - y)²)
        y_true_t = Tensor(np.asarray(y_true, dtype=float))
        diff = y_pred - y_true_t
        sq   = diff ** 2
        return sq.mean()

class BCE(Loss):
    def forward(self, y_pred: Tensor, y_true: np.ndarray) -> Tensor:
        # Pake asarray biar aman dari memoryview (ex. pandas/sklearn split)
        yp_data = np.asarray(y_pred.data, dtype=float)
        yt_data = np.asarray(y_true, dtype=float)

        # Clip biar gak kena log(0)
        p       = Tensor(np.clip(yp_data, _EPS, 1 - _EPS))
        y       = Tensor(yt_data)
        one_m_y = Tensor(1.0 - yt_data)
        one_m_p = Tensor(np.clip(1.0 - yp_data, _EPS, 1.0))

        # Rumus asli BCE: -mean(y*log(p) + (1-y)*log(1-p))
        pos = y * p.log()
        neg = one_m_y * one_m_p.log()
        loss = (pos + neg).mean()
        out = loss * (-1.0)

        # Karena ada clipping, autograd butuh bantuan 'hook' sedikit di sini
        _y_pred = y_pred

        def _bce_backward_hook():
            if _y_pred.requires_grad:
                _y_pred._ensure_grad()
                yp_arr = np.asarray(_y_pred.data, dtype=float)
                pv = np.clip(yp_arr, _EPS, 1 - _EPS)
                dL = (-((yt_data / pv) - (1.0 - yt_data) / (1.0 - pv))) / yp_arr.shape[0]
                _y_pred.grad += bce_out_grad_ref[0] * dL

        # Bikin node output beneran
        real_out = Tensor(out.data, _children=(_y_pred,), _op="bce")
        bce_out_grad_ref = [None]

        def _backward():
            bce_out_grad_ref[0] = real_out.grad
            _bce_backward_hook()

        real_out._backward = _backward
        return real_out

class CCE(Loss):
    def forward(self, y_pred: Tensor, y_true: np.ndarray) -> Tensor:
        # Categorical Cross-Entropy bwt multi-class
        yp_data = np.asarray(y_pred.data, dtype=float)
        yt_data = np.asarray(y_true, dtype=float)
        _y_pred = y_pred

        pv  = np.clip(yp_data, _EPS, 1.0)
        val = -float(np.mean(np.sum(yt_data * np.log(pv), axis=1)))

        out = Tensor(np.array([[val]]), _children=(y_pred,), _op="cce")

        def _backward():
            if _y_pred.requires_grad:
                _y_pred._ensure_grad()
                # Gradien normal CCE, tapi DenseLayer biasanya nge-override ini
                yp_arr = np.asarray(_y_pred.data, dtype=float)
                batch = yp_arr.shape[0]
                pv2 = np.clip(yp_arr, _EPS, 1.0)
                _y_pred.grad += out.grad.sum() * (-yt_data / pv2) / batch

        out._backward = _backward
        return out

LOSSES = {"mse": MSE, "bce": BCE, "cce": CCE}

def get_loss(name):
    if isinstance(name, Loss):
        return name
    try:
        return LOSSES[name.lower()]()
    except KeyError:
        raise ValueError(
            f"Gak ada loss '{name}'. Coba: {list(LOSSES)}"
        )
