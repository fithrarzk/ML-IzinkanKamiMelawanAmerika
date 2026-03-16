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
        
        # State khusus Adam
        self.m_W = np.zeros_like(self.W.data)
        self.v_W = np.zeros_like(self.W.data)
        self.m_b = np.zeros_like(self.b.data)
        self.v_b = np.zeros_like(self.b.data)
        self.t_adam = 0

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
        elif method == "xavier":
            # Xavier/Glorot Initialization (V = 2 / (n_in + n_out))
            rng    = np.random.default_rng(kw.get("seed"))
            std    = np.sqrt(2.0 / (self.n_in + self.n_out))
            W_data = rng.normal(0, std, (self.n_in, self.n_out))
            b_data = np.zeros((1, self.n_out)) # bias nol biasanya kalau xavier
        elif method == "he":
            # He Initialization (V = 2 / n_in)
            rng    = np.random.default_rng(kw.get("seed"))
            std    = np.sqrt(2.0 / self.n_in)
            W_data = rng.normal(0, std, (self.n_in, self.n_out))
            b_data = np.zeros((1, self.n_out)) # bias nol biasanya kalau He
        else:
            raise ValueError(
                f"Gak tau cara init '{method}'. Coba: zero | random_uniform | random_normal | xavier | he"
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

    def update(self, lr, regularization="none", lambda_=0.0, optimizer="sgd", beta1=0.9, beta2=0.999, epsilon=1e-8):
        # Update pake gradien yang udh diitung autograd
        reg = regularization.lower()
        if   reg == "l2": reg_term = lambda_ * self.W.data
        elif reg == "l1": reg_term = lambda_ * np.sign(self.W.data)
        else:             reg_term = 0.0

        grad_W = self.W.grad + reg_term
        grad_b = self.b.grad

        if optimizer.lower() == "adam":
            self.t_adam += 1
            
            self.m_W = beta1 * self.m_W + (1 - beta1) * grad_W
            self.v_W = beta2 * self.v_W + (1 - beta2) * (grad_W ** 2)
            m_hat_W = self.m_W / (1 - beta1 ** self.t_adam)
            v_hat_W = self.v_W / (1 - beta2 ** self.t_adam)
            self.W.data -= lr * m_hat_W / (np.sqrt(v_hat_W) + epsilon)
            
            self.m_b = beta1 * self.m_b + (1 - beta1) * grad_b
            self.v_b = beta2 * self.v_b + (1 - beta2) * (grad_b ** 2)
            m_hat_b = self.m_b / (1 - beta1 ** self.t_adam)
            v_hat_b = self.v_b / (1 - beta2 ** self.t_adam)
            self.b.data -= lr * m_hat_b / (np.sqrt(v_hat_b) + epsilon)
        else:
            self.W.data -= lr * grad_W
            self.b.data -= lr * grad_b

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
