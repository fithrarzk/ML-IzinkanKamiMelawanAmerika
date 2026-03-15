import numpy as np
from autograd import Tensor

class Activation:
    def forward(self, Z: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, Z: Tensor) -> Tensor:
        return self.forward(Z)

class Linear(Activation):
    def forward(self, Z: Tensor) -> Tensor:
        # Linear activation: f(z) = z
        # Cuma diterusin aja angkanya, dikali 1 biar direkam sama autograd graph
        return Z * 1.0

class ReLU(Activation):
    def forward(self, Z: Tensor) -> Tensor:
        # Rumus ReLU: max(0, Z)
        # akalin pake mask boolean: kalo > 0 jadi 1, sisanya 0
        import numpy as np
        mask = Tensor((Z.data > 0).astype(float))
        return Z * mask

class Sigmoid(Activation):
    def forward(self, Z: Tensor) -> Tensor:
        # Rumus Sigmoid: 1 / (1 + e^-Z)
        # Kita pake pangkat -1 karena __rtruediv__ (1.0 / Tensor) belum tentu disupport
        return (1.0 + (-Z).exp()) ** -1.0

class Tanh(Activation):
    def forward(self, Z: Tensor) -> Tensor:
        # Rumus Tanh: (e^Z - e^-Z) / (e^Z + e^-Z)
        # Turunannya bakal diturunin sempurna sama chain rule autograd
        e_z = Z.exp()
        e_neg_z = (-Z).exp()
        return (e_z - e_neg_z) / (e_z + e_neg_z)

class Softmax(Activation):
    def forward(self, Z: Tensor) -> Tensor:
        # Rumus Softmax: e^Z_i / sum(e^Z)
        # Kurangin max(Z) dulu biar stabil (gak NaN) pas dihitung exponensialnya
        import numpy as np
        z_max = Tensor(np.max(Z.data, axis=1, keepdims=True))
        e_z = (Z - z_max).exp()
        
        # pembagian dan rumusan _sum_ direkam sama autograd jadi _jacobian_ otomatis
        return e_z / e_z.sum(axis=1, keepdims=True)

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
            f"Gak kenal aktivasi '{name}'. Coba: {list(ACTIVATIONS)}"
        )
