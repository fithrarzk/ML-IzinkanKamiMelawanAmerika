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

        # W sama b dibungkus Tensor biar kena autograd
        self.W = Tensor(np.zeros((n_in, n_out)), requires_grad=True)
        self.b = Tensor(np.zeros((1, n_out)),    requires_grad=True)

        # Referensi buat bantu backward kalo pake softmax fusi
        self._A = None
        self._Z = None

        self._init_weights(init_method, **(init_params or {}))

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
                f"Gak tau cara init '{method}'. Coba: zero | random_uniform | random_normal"
            )

        # Isi data ke Tensor
        self.W.data = W_data.astype(np.float64)
        self.b.data = b_data.astype(np.float64)
        self.W.grad = np.zeros_like(self.W.data)
        self.b.grad = np.zeros_like(self.b.data)

    def forward(self, X: Tensor) -> Tensor:
        # Itung Z = XW + b, trs aktivasi
        self._Z = X.matmul(self.W) + self.b
        self._A = self.activation.forward(self._Z)
        return self._A

    def backward_softmax_fused(self, y_true: np.ndarray):
        # Trick biar Softmax + CCE lebih stabil pas backward
        batch = self._A.data.shape[0]
        fused_grad = (self._A.data - y_true) / batch
        self._Z._ensure_grad()
        self._Z.grad += fused_grad

    def update(self, lr, regularization="none", lambda_=0.0):
        # Update pake gradien yang udh diitung autograd
        reg = regularization.lower()
        if   reg == "l2": reg_term = lambda_ * self.W.data
        elif reg == "l1": reg_term = lambda_ * np.sign(self.W.data)
        else:             reg_term = 0.0

        self.W.data -= lr * (self.W.grad + reg_term)
        self.b.data -= lr * self.b.grad

    def zero_grad(self):
        # Bersihin gradien sblm batch baru
        self.W.zero_grad()
        self.b.zero_grad()

    # Properti ini biar kompatibel ama kode lama
    @property
    def dW(self):
        return self.W.grad if self.W.grad is not None else np.zeros_like(self.W.data)

    @property
    def db(self):
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
