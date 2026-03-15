"""
autograd.py — Mesin Automatic Differentiation

Terdiri dari dua lapisan:
  1. Value  — scalar autograd engine (ala micrograd Karpathy)
  2. Tensor — wrapper array numpy untuk operasi batched yang efisien

Cara kerja:
  - Setiap operasi forward membungkus hasilnya dalam Value/Tensor baru.
  - Setiap Value/Tensor menyimpan referensi ke "parent" nodes (_prev)
    dan sebuah fungsi _backward() yang mengimplementasikan chain rule
    untuk operasi tersebut.
  - Memanggil loss.backward() menjalankan topological sort pada
    computational graph, lalu mengakumulasikan gradien semua node
    secara otomatis — tanpa perlu menulis rumus turunan secara eksplisit.

Referensi: https://github.com/karpathy/micrograd
"""

import numpy as np


# =============================================================================
# Bagian 1: Value — scalar autograd engine
# =============================================================================

class Value:
    """
    Menyimpan sebuah nilai skalar dan gradiennya.
    Setiap operasi membangun computational graph secara otomatis.
    """

    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    # ------------------------------------------------------------------
    # Operator aritmetika (membangun graph)
    # ------------------------------------------------------------------

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Pangkat hanya mendukung int/float"
        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return Value(other) + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other ** -1

    def __rtruediv__(self, other):
        return Value(other) * self ** -1

    # ------------------------------------------------------------------
    # Fungsi matematika
    # ------------------------------------------------------------------

    def exp(self):
        val = np.exp(self.data)
        out = Value(val, (self,), "exp")

        def _backward():
            self.grad += val * out.grad

        out._backward = _backward
        return out

    def log(self):
        eps = 1e-15
        val = np.log(np.clip(self.data, eps, None))
        out = Value(val, (self,), "log")

        def _backward():
            self.grad += (1.0 / max(self.data, eps)) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        val = max(0.0, self.data)
        out = Value(val, (self,), "relu")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        val = np.tanh(self.data)
        out = Value(val, (self,), "tanh")

        def _backward():
            self.grad += (1 - val ** 2) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        val = 1.0 / (1.0 + np.exp(-np.clip(self.data, -500, 500)))
        out = Value(val, (self,), "sigmoid")

        def _backward():
            self.grad += val * (1 - val) * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Backward — topological sort + propagasi gradien
    # ------------------------------------------------------------------

    def backward(self):
        """
        Propagasi gradien dari node ini ke semua node di graph.
        Menggunakan topological sort agar setiap node diproses
        setelah semua penggunanya (parents) selesai.
        """
        topo = []
        visited = set()

        def build_topo(v):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def zero_grad(self):
        """Reset gradien ke 0."""
        topo = []
        visited = set()

        def build_topo(v):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        for v in topo:
            v.grad = 0.0

    def __repr__(self):
        return f"Value(data={self.data:.4g}, grad={self.grad:.4g})"


# =============================================================================
# Bagian 2: Tensor — batched numpy autograd wrapper
# =============================================================================

class Tensor:
    """
    Wrapper numpy array yang membangun computational graph untuk operasi
    batched (matmul, element-wise, reduction). Digunakan oleh DenseLayer,
    activations, dan losses.

    Prinsip kerjanya sama dengan Value:
      - Setiap operasi menghasilkan Tensor baru yang menyimpan _backward()
      - loss.backward() menyebarkan gradien ke semua parameter (W, b)
    """

    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        if isinstance(data, np.ndarray):
            self.data = data.astype(np.float64)
        else:
            self.data = np.array(data, dtype=np.float64)

        self.requires_grad = requires_grad or any(
            getattr(child, 'requires_grad', False) for child in _children
        )
        self.grad = np.zeros_like(self.data) if self.requires_grad else None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    # ------------------------------------------------------------------
    # Helper internal
    # ------------------------------------------------------------------

    def _ensure_grad(self):
        """Pastikan .grad terinisialisasi (untuk node intermediate)."""
        if self.grad is None:
            self.grad = np.zeros_like(self.data)

    # ------------------------------------------------------------------
    # Linear algebra
    # ------------------------------------------------------------------

    def matmul(self, other):
        """
        self @ other  (batched matrix multiply)
        Backward: chain rule matmul
          d_self  = out.grad @ other.data.T
          d_other = self.data.T @ out.grad
        """
        assert isinstance(other, Tensor)
        out = Tensor(self.data @ other.data, _children=(self, other), _op="matmul")

        def _backward():
            out._ensure_grad()
            if self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other._ensure_grad()
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def __add__(self, other):
        """
        element-wise add.
        Mendukung Tensor + Tensor (broadcasting untuk bias: (batch,out) + (1,out))
        """
        if isinstance(other, (int, float)):
            other = Tensor(np.full_like(self.data, other))
        assert isinstance(other, Tensor)
        out = Tensor(self.data + other.data, _children=(self, other), _op="+")

        def _backward():
            out._ensure_grad()
            if self.requires_grad:
                self._ensure_grad()
                # sum over axes yang di-broadcast
                grad = out.grad
                while grad.ndim > self.data.ndim:
                    grad = grad.sum(axis=0)
                for ax, (s, o) in enumerate(zip(self.data.shape, grad.shape)):
                    if s == 1 and o > 1:
                        grad = grad.sum(axis=ax, keepdims=True)
                self.grad += grad

            if other.requires_grad:
                other._ensure_grad()
                grad = out.grad
                while grad.ndim > other.data.ndim:
                    grad = grad.sum(axis=0)
                for ax, (s, o) in enumerate(zip(other.data.shape, grad.shape)):
                    if s == 1 and o > 1:
                        grad = grad.sum(axis=ax, keepdims=True)
                other.grad += grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(np.full_like(self.data, other))
        assert isinstance(other, Tensor)
        return self + (-other)

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(np.full_like(self.data, other))
        return other + (-self)

    def __neg__(self):
        out = Tensor(-self.data, _children=(self,), _op="neg")

        def _backward():
            out._ensure_grad()
            if self.requires_grad:
                self._ensure_grad()
                self.grad += -out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        """element-wise multiply"""
        if isinstance(other, (int, float)):
            other = Tensor(np.full_like(self.data, other))
        assert isinstance(other, Tensor)
        out = Tensor(self.data * other.data, _children=(self, other), _op="*")

        def _backward():
            out._ensure_grad()
            if self.requires_grad:
                self._ensure_grad()
                self.grad += other.data * out.grad
            if other.requires_grad:
                other._ensure_grad()
                other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return self * (1.0 / other)
        return self * other ** -1

    def __pow__(self, exp):
        assert isinstance(exp, (int, float))
        out = Tensor(self.data ** exp, _children=(self,), _op=f"**{exp}")

        def _backward():
            out._ensure_grad()
            if self.requires_grad:
                self._ensure_grad()
                self.grad += exp * (self.data ** (exp - 1)) * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Element-wise math ops (activation functions)
    # ------------------------------------------------------------------

    def exp(self):
        val = np.exp(np.clip(self.data, -500, 500))
        out = Tensor(val, _children=(self,), _op="exp")

        def _backward():
            out._ensure_grad()
            if self.requires_grad:
                self._ensure_grad()
                self.grad += val * out.grad

        out._backward = _backward
        return out

    def log(self):
        eps = 1e-15
        val = np.log(np.clip(self.data, eps, None))
        out = Tensor(val, _children=(self,), _op="log")

        def _backward():
            out._ensure_grad()
            if self.requires_grad:
                self._ensure_grad()
                self.grad += (1.0 / np.clip(self.data, eps, None)) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        mask = (self.data > 0).astype(np.float64)
        out = Tensor(self.data * mask, _children=(self,), _op="relu")

        def _backward():
            out._ensure_grad()
            if self.requires_grad:
                self._ensure_grad()
                self.grad += mask * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        val = np.tanh(self.data)
        out = Tensor(val, _children=(self,), _op="tanh")

        def _backward():
            out._ensure_grad()
            if self.requires_grad:
                self._ensure_grad()
                self.grad += (1 - val ** 2) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        val = 1.0 / (1.0 + np.exp(-np.clip(self.data, -500, 500)))
        out = Tensor(val, _children=(self,), _op="sigmoid")

        def _backward():
            out._ensure_grad()
            if self.requires_grad:
                self._ensure_grad()
                self.grad += val * (1 - val) * out.grad

        out._backward = _backward
        return out

    def softmax(self):
        """
        Softmax row-wise. Backward dikombinasikan dengan CCE loss
        menjadi (y_pred - y_true) / batch, sehingga tidak perlu
        Jacobian penuh (optimisasi seperti PyTorch).
        Catatan: backward ini hanya digunakan jika softmax berdiri sendiri.
        """
        e = np.exp(self.data - np.max(self.data, axis=1, keepdims=True))
        val = e / e.sum(axis=1, keepdims=True)
        out = Tensor(val, _children=(self,), _op="softmax")

        def _backward():
            out._ensure_grad()
            if self.requires_grad:
                self._ensure_grad()
                # Jacobian-vector product: dL/dZ = S*(dL/dA - sum(dL/dA * S, keepdim))
                s = val
                dot = np.sum(out.grad * s, axis=1, keepdims=True)
                self.grad += s * (out.grad - dot)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Reduction ops
    # ------------------------------------------------------------------

    def sum(self, axis=None, keepdims=False):
        val = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor(val, _children=(self,), _op="sum")

        def _backward():
            out._ensure_grad()
            if self.requires_grad:
                self._ensure_grad()
                grad = out.grad
                if axis is not None and not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                self.grad += np.broadcast_to(grad, self.data.shape).copy()

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # ------------------------------------------------------------------
    # Backward — topological sort dan propagasi gradien
    # ------------------------------------------------------------------

    def backward(self):
        """
        Propagasi gradien dari node ini (biasanya loss scalar)
        ke semua node yang terlibat dalam computational graph.
        """
        topo = []
        visited = set()

        def build_topo(v):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Seed: gradien output (loss scalar) adalah 1.0
        self._ensure_grad()
        self.grad = np.ones_like(self.data)

        for v in reversed(topo):
            v._backward()

    def zero_grad(self):
        """Reset gradien node ini ke nol."""
        if self.grad is not None:
            self.grad = np.zeros_like(self.data)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def shape(self):
        return self.data.shape

    @property
    def T(self):
        """Transpose — tidak membuat node graph (hanya view)."""
        return Tensor(self.data.T, requires_grad=self.requires_grad)

    def __repr__(self):
        return (
            f"Tensor(shape={self.shape}, "
            f"requires_grad={self.requires_grad}, "
            f"op='{self._op}')"
        )
