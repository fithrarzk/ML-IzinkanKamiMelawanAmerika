"""
ffnn.py — Feed-Forward Neural Network berbasis Autograd

Perubahan utama dari versi sebelumnya:
  - forward() mengonversi input X menjadi Tensor dan mengembalikan Tensor
  - backward() memanggil loss.backward() sehingga gradien menyebar otomatis
    ke seluruh computational graph (ke semua W dan b di setiap layer)
  - zero_grad() mereset semua gradien sebelum tiap batch
  - update_weights() membaca .grad dari Tensor W dan b (sudah diisi oleh autograd)
  - loss_fn.forward() sekarang mengembalikan Tensor (bukan float)
"""

import numpy as np
from autograd import Tensor
from layer import DenseLayer
from losses import get_loss, LOSSES


class FFNN:
    def __init__(self, layer_configs: list[dict]):
        self.layer_configs = layer_configs
        self.layers: list[DenseLayer] = [
            DenseLayer(
                n_in=cfg["n_in"],
                n_out=cfg["n_out"],
                activation=cfg.get("activation", "linear"),
                init_method=cfg.get("init_method", "random_normal"),
                init_params=cfg.get("init_params", {}),
            )
            for cfg in layer_configs
        ]

        self.loss_fn      = None
        self.lr           = 0.01
        self.regularization = "none"
        self.lambda_      = 0.0
        self.history      = {"train_loss": [], "val_loss": []}

    def compile(self, loss="mse", lr=0.01, regularization="none", lambda_=0.0):
        self.loss_fn        = get_loss(loss)
        self.lr             = lr
        self.regularization = regularization
        self.lambda_        = lambda_

        # Tandai layer output jika merupakan Softmax + CCE (fused backward)
        last     = self.layers[-1]
        is_cce    = (loss.lower() == "cce") if isinstance(loss, str) else False
        is_softmax = last.activation.__class__.__name__.lower() == "softmax"
        last.is_softmax_output = is_cce and is_softmax

    # ------------------------------------------------------------------
    # Regularisasi
    # ------------------------------------------------------------------

    def _reg_penalty(self) -> float:
        reg = self.regularization.lower()
        if reg == "l2":
            return self.lambda_ * sum(np.sum(l.W.data ** 2) for l in self.layers)
        if reg == "l1":
            return self.lambda_ * sum(np.sum(np.abs(l.W.data)) for l in self.layers)
        return 0.0

    # ------------------------------------------------------------------
    # Forward — membangun computational graph
    # ------------------------------------------------------------------

    def forward(self, X: np.ndarray) -> Tensor:
        """
        Jalankan forward pass.
        Input X (numpy) dikonversi ke Tensor tanpa grad (data saja).
        Hasilnya adalah Tensor yang sudah terhubung ke W dan b tiap layer.
        """
        A = Tensor(X)           # input: tidak perlu grad
        for layer in self.layers:
            A = layer.forward(A)
        return A                 # Tensor output

    # ------------------------------------------------------------------
    # Zero grad — wajib dipanggil sebelum tiap batch
    # ------------------------------------------------------------------

    def zero_grad(self):
        """Reset semua gradien W dan b ke nol."""
        for layer in self.layers:
            layer.zero_grad()

    # ------------------------------------------------------------------
    # Backward — autograd menyebarkan gradien ke seluruh graph
    # ------------------------------------------------------------------

    def backward(self, y_pred: Tensor, y_true: np.ndarray):
        """
        Hitung loss, lalu panggil loss.backward().
        Autograd akan menyebarkan gradien secara otomatis ke semua W dan b.

        Jika layer output adalah Softmax+CCE, kita inject gradien fused
        (ŷ - y)/batch langsung ke _Z output layer sebelum backward mengalir ke W.
        """
        loss_tensor = self.loss_fn.forward(y_pred, y_true)

        if self.layers[-1].is_softmax_output:
            # Fused Softmax+CCE backward: inject gradien ke pre-activation
            # lalu backward mengalir dari _Z ke W dan b
            self.layers[-1].backward_softmax_fused(y_true)
            # Alirkan dari _Z ke W, b, dan input layer sebelumnya
            self._backprop_from_z(len(self.layers) - 1)
        else:
            # Autograd murni: panggil backward pada loss Tensor
            loss_tensor.backward()

        return loss_tensor

    def _backprop_from_z(self, from_layer_idx: int):
        """
        Untuk mode fused softmax+CCE: gradien sudah di-inject ke _Z
        pada layer from_layer_idx. Kita perlu mengalirkan gradien ini
        ke W, b, dan ke input layer (dA_prev), lalu teruskan ke layer
        sebelumnya melalui autograd.
        """
        for i in range(from_layer_idx, -1, -1):
            layer = self.layers[i]
            dZ = layer._Z.grad           # sudah diisi (fused inject atau dari layer di atas)
            X  = layer._Z._prev          # Tensor input ke layer ini

            # Ambil X (input layer) — ia adalah _A dari layer sebelumnya atau input asli
            # Kita cari node X dari graph _Z
            # Struktur: _Z = X.matmul(W) + b
            # _Z._prev = {add_node}  → add_node._prev = {matmul_node, b}
            # matmul_node._prev = {X_tensor, W}
            # Kita trigger _backward() pada sub-graph ini secara manual
            # dengan mendelegasikan ke topological backward dari _Z ke W dan b saja.
            if dZ is None:
                continue

            # Hitung dW, db dari dZ
            # Dapatkan X (input ke layer) dari graph matmul
            # _Z berasal dari: add(matmul(X, W), b)
            # Jalankan backward sub-graph dari _Z
            layer._Z._ensure_grad()
            self._backward_subgraph(layer._Z, stop_at_inputs=True)

    def _backward_subgraph(self, root_tensor, stop_at_inputs=True):
        """
        Jalankan topological backward hanya untuk sub-graph dari root_tensor.
        Ini dipanggil untuk mode fused softmax+CCE setelah gradien diinjeksi
        ke _Z layer output.
        """
        topo    = []
        visited = set()

        def build(v):
            if id(v) not in visited:
                visited.add(id(v))
                for c in v._prev:
                    build(c)
                topo.append(v)

        build(root_tensor)
        for v in reversed(topo):
            v._backward()

    # ------------------------------------------------------------------
    # Update bobot
    # ------------------------------------------------------------------

    def update_weights(self):
        for layer in self.layers:
            layer.update(self.lr, self.regularization, self.lambda_)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        verbose: int = 1,
    ) -> dict:
        self.history = {"train_loss": [], "val_loss": []}
        n = X.shape[0]

        for epoch in range(1, epochs + 1):
            idx = np.random.permutation(n)
            Xs, ys = X[idx], y[idx]
            batch_losses = []

            for start in range(0, n, batch_size):
                Xb, yb = Xs[start: start + batch_size], ys[start: start + batch_size]

                # 1. Reset semua gradien (penting! agar tidak akumulasi antar batch)
                self.zero_grad()

                # 2. Forward — bangun computational graph
                y_pred = self.forward(Xb)

                # 3. Backward — autograd menyebarkan gradien
                loss_tensor = self.backward(y_pred, yb)

                # 4. Update bobot menggunakan gradien dari Tensor.grad
                self.update_weights()

                batch_losses.append(float(loss_tensor.data.flat[0]))

            train_loss = float(np.mean(batch_losses)) + self._reg_penalty()
            self.history["train_loss"].append(train_loss)

            if X_val is not None and y_val is not None:
                with_no_grad = self.forward(X_val)
                val_loss_t   = self.loss_fn.forward(with_no_grad, y_val)
                self.history["val_loss"].append(
                    float(val_loss_t.data.flat[0]) + self._reg_penalty()
                )

            if verbose and epoch % verbose == 0:
                msg = f"Epoch {epoch:>4}/{epochs}  loss: {train_loss:.6f}"
                if self.history["val_loss"]:
                    msg += f"  val_loss: {self.history['val_loss'][-1]:.6f}"
                print(msg)

        return self.history

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Kembalikan numpy array (bukan Tensor)."""
        return self.forward(X).data

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, filepath: str):
        arrays = {}
        for i, layer in enumerate(self.layers):
            arrays[f"W_{i}"] = layer.W.data
            arrays[f"b_{i}"] = layer.b.data

        config_arr = np.empty(len(self.layer_configs), dtype=object)
        for i, cfg in enumerate(self.layer_configs):
            config_arr[i] = cfg

        arrays["layer_configs"]   = config_arr
        arrays["loss_fn_name"]    = np.array([self.loss_fn.__class__.__name__])
        arrays["lr"]              = np.array([self.lr])
        arrays["regularization"]  = np.array([self.regularization])
        arrays["lambda_"]         = np.array([self.lambda_])

        np.savez(filepath, **arrays)
        print(f"Model saved → '{filepath}'")

    @classmethod
    def load(cls, filepath: str) -> "FFNN":
        data        = np.load(filepath, allow_pickle=True)
        layer_configs = list(data["layer_configs"])
        model       = cls(layer_configs)
        for i, layer in enumerate(model.layers):
            layer.set_params(data[f"W_{i}"], data[f"b_{i}"])
        loss_map = {v.__name__: k for k, v in LOSSES.items()}
        model.compile(
            loss           = loss_map.get(str(data["loss_fn_name"][0]), "mse"),
            lr             = float(data["lr"][0]),
            regularization = str(data["regularization"][0]),
            lambda_        = float(data["lambda_"][0]),
        )
        print(f"Model loaded = '{filepath}'")
        return model

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self):
        total = 0
        for i, layer in enumerate(self.layers):
            p = layer.W.data.size + layer.b.data.size
            total += p
            print(f"  Layer {i+1}: {layer}  [{p:,} params]")
        print("-" * 52)
        print(f"  Total params: {total:,}")
        print("=" * 52)

    def __repr__(self):
        return f"FFNN(layers={len(self.layers)})"
