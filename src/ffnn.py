import numpy as np
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

        self.loss_fn = None
        self.lr = 0.01
        self.regularization = "none"
        self.lambda_ = 0.0
        self.history = {"train_loss": [], "val_loss": []}

    def compile(self, loss="mse", lr=0.01, regularization="none", lambda_=0.0):
        self.loss_fn = get_loss(loss)
        self.lr = lr
        self.regularization = regularization
        self.lambda_ = lambda_

        last = self.layers[-1]
        is_cce = (loss.lower() == "cce") if isinstance(loss, str) else False
        is_softmax = last.activation.__class__.__name__.lower() == "softmax"
        last.is_softmax_output = is_cce and is_softmax

    def _reg_penalty(self) -> float:
        reg = self.regularization.lower()
        if reg == "l2":
            return self.lambda_ * sum(np.sum(l.W**2) for l in self.layers)
        if reg == "l1":
            return self.lambda_ * sum(np.sum(np.abs(l.W)) for l in self.layers)
        return 0.0

    def forward(self, X: np.ndarray) -> np.ndarray:
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray):
        dA = self.loss_fn.backward(y_pred, y_true)
        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    def update_weights(self):
        for layer in self.layers:
            layer.update(self.lr, self.regularization, self.lambda_)

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
                Xb, yb = Xs[start : start + batch_size], ys[start : start + batch_size]
                y_pred = self.forward(Xb)
                self.backward(y_pred, yb)
                self.update_weights()
                batch_losses.append(self.loss_fn.forward(y_pred, yb))

            train_loss = float(np.mean(batch_losses)) + self._reg_penalty()
            self.history["train_loss"].append(train_loss)

            if X_val is not None and y_val is not None:
                val_pred = self.forward(X_val)
                self.history["val_loss"].append(
                    self.loss_fn.forward(val_pred, y_val) + self._reg_penalty()
                )

            if verbose and epoch % verbose == 0:
                msg = f"Epoch {epoch:>4}/{epochs}  loss: {train_loss:.6f}"
                if self.history["val_loss"]:
                    msg += f"  val_loss: {self.history['val_loss'][-1]:.6f}"
                print(msg)

        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def save(self, filepath: str):
        arrays = {}
        for i, layer in enumerate(self.layers):
            arrays[f"W_{i}"] = layer.W
            arrays[f"b_{i}"] = layer.b

        config_arr = np.empty(len(self.layer_configs), dtype=object)
        for i, cfg in enumerate(self.layer_configs):
            config_arr[i] = cfg

        arrays["layer_configs"] = config_arr
        arrays["loss_fn_name"] = np.array([self.loss_fn.__class__.__name__])
        arrays["lr"] = np.array([self.lr])
        arrays["regularization"] = np.array([self.regularization])
        arrays["lambda_"] = np.array([self.lambda_])

        np.savez(filepath, **arrays)
        print(f"Model saved → '{filepath}'")

    @classmethod
    def load(cls, filepath: str) -> "FFNN":
        data = np.load(filepath, allow_pickle=True)
        layer_configs = list(data["layer_configs"])
        model = cls(layer_configs)
        for i, layer in enumerate(model.layers):
            layer.set_params(data[f"W_{i}"], data[f"b_{i}"])
        loss_map = {v.__name__: k for k, v in LOSSES.items()}
        model.compile(
            loss=loss_map.get(str(data["loss_fn_name"][0]), "mse"),
            lr=float(data["lr"][0]),
            regularization=str(data["regularization"][0]),
            lambda_=float(data["lambda_"][0]),
        )
        print(f"Model loaded = '{filepath}'")
        return model

    def summary(self):
        total = 0
        for i, layer in enumerate(self.layers):
            p = layer.W.size + layer.b.size
            total += p
            print(f"  Layer {i+1}: {layer}  [{p:,} params]")
        print("-" * 52)
        print(f"  Total params: {total:,}")
        print("=" * 52)

    def __repr__(self):
        return f"FFNN(layers={len(self.layers)})"
