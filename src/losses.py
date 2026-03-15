"""
losses.py — Loss functions berbasis Autograd

Setiap loss mengimplementasikan forward() yang menerima Tensor y_pred
dan array numpy y_true, lalu membangun computational graph secara otomatis.
Hasilnya adalah Tensor scalar. Memanggil loss.backward() akan menyebarkan
gradien ke seluruh graph — tidak ada lagi metode backward() yang ditulis manual.
"""

import numpy as np
from autograd import Tensor

_EPS = 1e-15


class Loss:
    def forward(self, y_pred: Tensor, y_true: np.ndarray) -> Tensor:
        raise NotImplementedError

    def __call__(self, y_pred: Tensor, y_true: np.ndarray) -> Tensor:
        return self.forward(y_pred, y_true)


class MSE(Loss):
    """
    Mean Squared Error:  L = mean((ŷ - y)²)

    Graph yang dibangun:
      diff = y_pred - y_true_tensor
      sq   = diff ** 2
      loss = sq.mean()

    Backward diturunkan otomatis:
      dL/dŷ = 2*(ŷ - y) / n
    """

    def forward(self, y_pred: Tensor, y_true: np.ndarray) -> Tensor:
        y_true_t = Tensor(y_true)          # constant, requires_grad=False
        diff = y_pred - y_true_t
        sq   = diff ** 2
        return sq.mean()


class BCE(Loss):
    """
    Binary Cross-Entropy:
      L = -mean( y*log(ŷ) + (1-y)*log(1-ŷ) )

    Graph yang dibangun via Tensor log() dan mul ops.
    Backward diturunkan otomatis:
      dL/dŷ = (-(y/ŷ) + (1-y)/(1-ŷ)) / n
    """

    def forward(self, y_pred: Tensor, y_true: np.ndarray) -> Tensor:
        p      = Tensor(np.clip(y_pred.data, _EPS, 1 - _EPS))
        y      = Tensor(y_true)
        one_m_y = Tensor(1.0 - y_true)
        one_m_p = Tensor(np.clip(1.0 - y_pred.data, _EPS, 1.0))

        # -( y*log(p) + (1-y)*log(1-p) ).mean()
        pos = y * p.log()
        neg = one_m_y * one_m_p.log()
        loss = (pos + neg).mean()

        # Negate: loss = -mean(...)
        out = loss * (-1.0)

        # Kaitkan y_pred ke graph sehingga backward bisa mengalir ke y_pred.grad
        # Kita perlu mengerjakan backward ke y_pred secara eksplisit
        # karena kita meng-clip data di atas (p berdiri sendiri dari y_pred).
        # Implementasi manual backward yang benar untuk BCE:
        _y_pred = y_pred
        _y_true = y_true

        def _bce_backward_hook():
            if _y_pred.requires_grad:
                _y_pred._ensure_grad()
                pv = np.clip(_y_pred.data, _EPS, 1 - _EPS)
                dL = (-((_y_true / pv) - (1 - _y_true) / (1 - pv))) / _y_pred.data.shape[0]
                _y_pred.grad += out.grad * dL

        # Tambahkan y_pred sebagai child dari out dengan custom backward
        real_out = Tensor(out.data, _children=(_y_pred,), _op="bce")
        bce_out_grad_ref = [None]

        def _backward():
            bce_out_grad_ref[0] = real_out.grad
            _bce_backward_hook()

        real_out._backward = _backward
        return real_out


class CCE(Loss):
    """
    Categorical Cross-Entropy:
      L = -mean( sum_k( y_k * log(ŷ_k) ) )

    Ketika dipakai bersama Softmax (flag is_softmax_output=True pada DenseLayer),
    backward difusikan menjadi (ŷ - y)/batch — seperti di PyTorch.
    Dalam kasus umum, backward diturunkan otomatis melalui Tensor.log().
    """

    def forward(self, y_pred: Tensor, y_true: np.ndarray) -> Tensor:
        _y_pred = y_pred
        _y_true = y_true

        pv  = np.clip(y_pred.data, _EPS, 1.0)
        val = -float(np.mean(np.sum(y_true * np.log(pv), axis=1)))

        out = Tensor(np.array([[val]]), _children=(y_pred,), _op="cce")

        def _backward():
            if _y_pred.requires_grad:
                _y_pred._ensure_grad()
                # Gradien standar CCE terhadap ŷ (sebelum softmax):
                # dL/dŷ_k = -y_k / ŷ_k / batch
                # (jika softmax di-fuse, DenseLayer akan meng-override ini menjadi ŷ-y)
                batch = _y_pred.data.shape[0]
                pv2 = np.clip(_y_pred.data, _EPS, 1.0)
                _y_pred.grad += out.grad.sum() * (-_y_true / pv2) / batch

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
            f"Unknown loss '{name}'. Choose from {list(LOSSES)}"
        )
