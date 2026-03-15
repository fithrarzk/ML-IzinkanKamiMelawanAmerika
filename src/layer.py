"""
layer.py — DenseLayer berbasis Autograd

Perubahan utama dari versi sebelumnya:
  - self.W dan self.b sekarang adalah Tensor dengan requires_grad=True
  - forward() membangun computational graph: Z = X.matmul(W) + b, A = activation(Z)
  - backward() tidak lagi menerima gradien secara manual; cukup panggil
    loss.backward() di level FFNN, lalu baca .grad dari Tensor W dan b
  - update() membaca W.grad dan b.grad yang sudah diisi oleh autograd
  - zero_grad() mereset W.grad dan b.grad ke nol sebelum tiap batch
"""

import numpy as np
from autograd import Tensor
from activations import get_activation


class DenseLayer:
    def __init__(
        self,
        n_in,
        n_out,
        activation="linear",
        init_method="random_normal",
        init_params=None,
        is_softmax_output=False,
    ):
        self.n_in   = n_in
        self.n_out  = n_out
        self.activation      = get_activation(activation)
        self.is_softmax_output = is_softmax_output

        # W dan b adalah Tensor dengan requires_grad=True
        # — inilah yang membuat autograd bisa mengalirkan gradien ke sini
        self.W = Tensor(np.zeros((n_in, n_out)), requires_grad=True)
        self.b = Tensor(np.zeros((1, n_out)),    requires_grad=True)

        # Simpan referensi ke output layer untuk keperluan backward softmax-fused
        self._A = None   # Tensor output setelah aktivasi
        self._Z = None   # Tensor pre-aktivasi

        self._init_weights(init_method, **(init_params or {}))

    # ------------------------------------------------------------------
    # Inisialisasi bobot
    # ------------------------------------------------------------------

    def _init_weights(self, method, **kw):
        method = method.lower()
        if method == "zero":
            W_data = np.zeros((self.n_in, self.n_out))
            b_data = np.zeros((1, self.n_out))

        elif method == "random_uniform":
            rng    = np.random.default_rng(kw.get("seed"))
            W_data = rng.uniform(kw.get("low", 0), kw.get("high", 1), (self.n_in, self.n_out))
            b_data = rng.uniform(kw.get("low", 0), kw.get("high", 1), (1, self.n_out))

        elif method == "random_normal":
            rng    = np.random.default_rng(kw.get("seed"))
            W_data = rng.normal(kw.get("mean", 0), kw.get("std", 1), (self.n_in, self.n_out))
            b_data = rng.normal(kw.get("mean", 0), kw.get("std", 1), (1, self.n_out))

        else:
            raise ValueError(
                f"Unknown init_method '{method}'. Choose: zero | random_uniform | random_normal"
            )

        # Update data di Tensor (grad tetap 0)
        self.W.data = W_data.astype(np.float64)
        self.b.data = b_data.astype(np.float64)
        self.W.grad = np.zeros_like(self.W.data)
        self.b.grad = np.zeros_like(self.b.data)

    # ------------------------------------------------------------------
    # Forward pass — membangun computational graph
    # ------------------------------------------------------------------

    def forward(self, X: Tensor) -> Tensor:
        """
        Z = X @ W + b
        A = activation(Z)

        Setiap operasi di sini (matmul, add, activation) mendaftarkan
        _backward() masing-masing ke dalam graph secara otomatis.
        """
        self._Z = X.matmul(self.W) + self.b
        self._A = self.activation.forward(self._Z)
        return self._A

    # ------------------------------------------------------------------
    # Backward pass — didelegasikan sepenuhnya ke autograd
    # ------------------------------------------------------------------

    def backward_softmax_fused(self, y_true: np.ndarray):
        """
        Optimasi khusus untuk layer output dengan Softmax + CCE:
        dL/dZ = (ŷ - y) / batch  (fused gradient)

        Ini identik dengan yang dilakukan PyTorch saat menggunakan
        CrossEntropyLoss + Softmax output.

        Dipanggil oleh FFNN sebelum loss.backward() jika is_softmax_output=True.
        Mode ini meng-inject gradien ke self._Z.grad secara langsung,
        lalu backward akan mengalir ke W dan b.
        """
        batch = self._A.data.shape[0]
        fused_grad = (self._A.data - y_true) / batch
        self._Z._ensure_grad()
        self._Z.grad += fused_grad

    # ------------------------------------------------------------------
    # Update bobot setelah backward
    # ------------------------------------------------------------------

    def update(self, lr, regularization="none", lambda_=0.0):
        """
        Update W dan b menggunakan gradien yang sudah diisi oleh autograd.
        """
        reg = regularization.lower()
        if   reg == "l2": reg_term = lambda_ * self.W.data
        elif reg == "l1": reg_term = lambda_ * np.sign(self.W.data)
        else:             reg_term = 0.0

        self.W.data -= lr * (self.W.grad + reg_term)
        self.b.data -= lr * self.b.grad

    def zero_grad(self):
        """Reset gradien W dan b ke nol sebelum batch berikutnya."""
        self.W.zero_grad()
        self.b.zero_grad()

    # ------------------------------------------------------------------
    # Utilities (kompatibilitas dengan utils.py)
    # ------------------------------------------------------------------

    @property
    def dW(self):
        """Kompatibilitas mundur: akses gradien W seperti sebelumnya."""
        return self.W.grad if self.W.grad is not None else np.zeros_like(self.W.data)

    @property
    def db(self):
        """Kompatibilitas mundur: akses gradien b seperti sebelumnya."""
        return self.b.grad if self.b.grad is not None else np.zeros_like(self.b.data)

    def get_params(self):
        return {"W": self.W.data, "b": self.b.data}

    def get_gradients(self):
        return {"dW": self.dW, "db": self.db}

    def set_params(self, W, b):
        self.W.data = W.copy().astype(np.float64)
        self.b.data = b.copy().astype(np.float64)
        self.W.grad = np.zeros_like(self.W.data)
        self.b.grad = np.zeros_like(self.b.data)

    def __repr__(self):
        return (
            f"DenseLayer({self.n_in} → {self.n_out}, "
            f"activation={self.activation.__class__.__name__})"
        )
