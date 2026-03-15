"""
activations.py — Fungsi aktivasi berbasis Autograd

Setiap activation mengimplementasikan forward() yang menerima Tensor
dan mengembalikan Tensor. Karena Tensor membangun computational graph
secara otomatis, tidak ada lagi metode backward() yang ditulis manual
— gradien akan diturunkan secara otomatis oleh mesin autograd.
"""

from autograd import Tensor


class Activation:
    def forward(self, Z: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, Z: Tensor) -> Tensor:
        return self.forward(Z)


class Linear(Activation):
    """f(z) = z  →  f'(z) = 1  (ditangani autograd via identity)"""

    def forward(self, Z: Tensor) -> Tensor:
        # Buat node baru dengan _backward yang meneruskan gradien apa adanya
        import numpy as np
        out = Tensor(Z.data.copy(), _children=(Z,), _op="linear")

        def _backward():
            if Z.requires_grad:
                Z._ensure_grad()
                Z.grad += out.grad

        out._backward = _backward
        return out


class ReLU(Activation):
    """f(z) = max(0, z)  →  autograd via Tensor.relu()"""

    def forward(self, Z: Tensor) -> Tensor:
        return Z.relu()


class Sigmoid(Activation):
    """f(z) = 1/(1+e^-z)  →  autograd via Tensor.sigmoid()"""

    def forward(self, Z: Tensor) -> Tensor:
        return Z.sigmoid()


class Tanh(Activation):
    """f(z) = tanh(z)  →  autograd via Tensor.tanh()"""

    def forward(self, Z: Tensor) -> Tensor:
        return Z.tanh()


class Softmax(Activation):
    """
    f(z)_i = exp(z_i) / sum_j(exp(z_j))

    Backward ditangani autograd via Tensor.softmax() yang mengimplementasikan
    Jacobian-vector product:  dL/dZ = S * (dL/dA - sum(dL/dA * S, axis=1))

    Ketika dipakai bersama CCE loss, FFNN menerapkan optimasi fusi
    (softmax + CCE backward = y_pred - y_true), seperti yang dilakukan
    oleh PyTorch (flag is_softmax_output di DenseLayer).
    """

    def forward(self, Z: Tensor) -> Tensor:
        return Z.softmax()


ACTIVATIONS = {
    "linear":  Linear,
    "relu":    ReLU,
    "sigmoid": Sigmoid,
    "tanh":    Tanh,
    "softmax": Softmax,
}


def get_activation(name):
    if isinstance(name, Activation):
        return name
    try:
        return ACTIVATIONS[name.lower()]()
    except KeyError:
        raise ValueError(
            f"Unknown activation '{name}'. Choose from {list(ACTIVATIONS)}"
        )
