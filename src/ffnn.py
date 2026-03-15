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

    #regularisasi
    def _reg_penalty(self) -> float:
        reg = self.regularization.lower()
        if reg == "l2":
            return self.lambda_ * sum(np.sum(l.W.data ** 2) for l in self.layers)
        if reg == "l1":
            return self.lambda_ * sum(np.sum(np.abs(l.W.data)) for l in self.layers)
        return 0.0

    # forward, membangun computational graph
    def forward(self, X: np.ndarray) -> Tensor:
        A = Tensor(X)           # input: tidak perlu grad
        for layer in self.layers:
            A = layer.forward(A)
        return A                 # Tensor output

    # zero grad — wajib dipanggil sebelum tiap batch
    def zero_grad(self):
        """Reset semua gradien W dan b ke nol."""
        for layer in self.layers:
            layer.zero_grad()


    # backward — autograd menyebarkan gradien ke seluruh graph
    def backward(self, y_pred: Tensor, y_true: np.ndarray):
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
        for i in range(from_layer_idx, -1, -1):
            layer = self.layers[i]
            dZ = layer._Z.grad           # sudah diisi (fused inject atau dari layer di atas)
            X  = layer._Z._prev          # Tensor input ke layer ini
            if dZ is None:
                continue

            layer._Z._ensure_grad()
            self._backward_subgraph(layer._Z, stop_at_inputs=True)

    def _backward_subgraph(self, root_tensor, stop_at_inputs=True):
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

    # Update bobot
    def update_weights(self, lr, regularization="none", lambda_=0.0):
        # Update semua parameter di tiap layer
        for layer in self.layers:
            layer.update(lr, regularization, lambda_)

    def zero_grad(self):
        # Bersihin gradien sebelum forward/backward baru
        for layer in self.layers:
            layer.zero_grad()

    def fit(
        self,
        X,
        y,
        epochs=100,
        batch_size=32,
        X_val=None,
        y_val=None,
        lr=0.01,
        regularization="none",
        lambda_=0.0,
        verbose=1,
    ):
        self.history = {"train_loss": [], "val_loss": []}
        n_samples = X.shape[0]

        for epoch in range(epochs):
            # Shuffle data biar variatif tiap epoch
            indices = np.random.permutation(n_samples)
            Xs, ys = X[indices], y[indices]

            epoch_losses = []

            # Training per batch
            for i in range(0, n_samples, batch_size):
                Xb, yb = Xs[i : i + batch_size], ys[i : i + batch_size]

                # Reset gradien
                self.zero_grad()

                # Forward
                y_pred = self.forward(Xb)

                # Loss buat dipantau (pake data aja, gak butuh graph di sini)
                # Tujuannya biar history gak numpuk memori graph
                loss_val = float(self.loss_fn.forward(y_pred, yb).data.sum())
                epoch_losses.append(loss_val)

                # Backward (ini yg build graph & itung grad)
                self.backward(y_pred, yb)

                # Update bobot
                self.update_weights(lr, regularization, lambda_)

            avg_train_loss = np.mean(epoch_losses)
            self.history["train_loss"].append(avg_train_loss)

            # Validasi kalo ada
            val_msg = ""
            if X_val is not None and y_val is not None:
                y_val_pred = self.forward(X_val)
                avg_val_loss = float(self.loss_fn.forward(y_val_pred, y_val).data.sum())
                self.history["val_loss"].append(avg_val_loss)
                val_msg = f", Val Loss: {avg_val_loss:.4f}"

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}{val_msg}")

        return self.history

    # Predict
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Kembalikan numpy array (bukan Tensor)."""
        return self.forward(X).data

    # Save / Load
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

    # Summary
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
