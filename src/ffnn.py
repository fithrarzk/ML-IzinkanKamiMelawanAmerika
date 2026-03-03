import numpy as np
from layer import DenseLayer
from losses import get_loss
from activations import get_activation


class FFNN:
    def __init__(self, layer_configs: list[dict]):
        self.layer_configs = layer_configs
        self.layers: list[DenseLayer] = [
            DenseLayer(
                n_in        = cfg['n_in'],
                n_out       = cfg['n_out'],
                activation  = cfg.get('activation', 'linear'),
                init_method = cfg.get('init_method', 'random_normal'),
                init_params = cfg.get('init_params', {}),
            )
            for cfg in layer_configs
        ]

        # compile() fills these
        self.loss_fn       = None
        self.lr            = None
        self.regularization = 'none'
        self.lambda_       = 0.0
        self.history       = {'train_loss': [], 'val_loss': []}


    def compile(self, loss='mse', lr=0.01, regularization='none', lambda_=0.0):
        self.loss_fn        = get_loss(loss)
        self.lr             = lr
        self.regularization = regularization
        self.lambda_        = lambda_

        # Mark output layer for fused Softmax+CCE gradient
        last = self.layers[-1]
        is_cce     = loss.lower() == 'cce' if isinstance(loss, str) else isinstance(self.loss_fn, type(get_loss('cce')))
        is_softmax = last.activation.__class__.__name__.lower() == 'softmax'
        last.is_softmax_output = is_cce and is_softmax

    def forward(self, X: np.ndarray) -> np.ndarray:
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray):
        dA = self.loss_fn.backward(y_pred, y_true)
        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    def _update_weights(self):
        for layer in self.layers:
            layer.update(self.lr, self.regularization, self.lambda_)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 32, 
            X_val: np.ndarray = None, y_val: np.ndarray = None, verbose: int = 1) -> dict:
        self.history = {'train_loss': [], 'val_loss': []}
        n_samples = X.shape[0]

        for epoch in range(1, epochs + 1):
            # Shuffle training data
            idx = np.random.permutation(n_samples)
            X_shuffled, y_shuffled = X[idx], y[idx]

            epoch_loss = 0.0
            n_batches  = 0

            for start in range(0, n_samples, batch_size):
                Xb = X_shuffled[start:start + batch_size]
                yb = y_shuffled[start:start + batch_size]

                y_pred    = self.forward(Xb)
                batch_loss = self.loss_fn.forward(y_pred, yb)
                self.backward(y_pred, yb)
                self._update_weights()

                epoch_loss += batch_loss
                n_batches  += 1

            train_loss = epoch_loss / n_batches
            self.history['train_loss'].append(train_loss)

            if X_val is not None and y_val is not None:
                val_pred = self.forward(X_val)
                val_loss = self.loss_fn.forward(val_pred, y_val)
                self.history['val_loss'].append(val_loss)

            if verbose and (epoch % verbose == 0 or epoch == 1):
                msg = f"Epoch {epoch:>4}/{epochs}  loss: {train_loss:.6f}"
                if self.history['val_loss']:
                    msg += f"  val_loss: {self.history['val_loss'][-1]:.6f}"
                print(msg)

        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)


    def save(self, filepath: str):
        arrays = {}
        for i, layer in enumerate(self.layers):
            arrays[f'W_{i}'] = layer.W
            arrays[f'b_{i}'] = layer.b

        # Store architecture as a serialisable array of dicts via object array
        config_arr = np.empty(len(self.layer_configs), dtype=object)
        for i, cfg in enumerate(self.layer_configs):
            config_arr[i] = cfg

        arrays['layer_configs']    = config_arr
        arrays['loss_fn_name']     = np.array([self.loss_fn.__class__.__name__])
        arrays['lr']               = np.array([self.lr])
        arrays['regularization']   = np.array([self.regularization])
        arrays['lambda_']          = np.array([self.lambda_])

        np.savez(filepath, **arrays)
        print(f"Model saved to '{filepath}'.")

    @classmethod
    def load(cls, filepath: str) -> 'FFNN':
        data = np.load(filepath, allow_pickle=True)

        layer_configs = list(data['layer_configs'])
        model = cls(layer_configs)

        for i, layer in enumerate(model.layers):
            layer.set_params(data[f'W_{i}'], data[f'b_{i}'])

        loss_name = str(data['loss_fn_name'][0])
        from losses import LOSSES
        loss_map  = {v.__name__: k for k, v in LOSSES.items()}
        lr            = float(data['lr'][0])
        regularization = str(data['regularization'][0])
        lambda_       = float(data['lambda_'][0])

        model.compile(
            loss           = loss_map.get(loss_name, 'mse'),
            lr             = lr,
            regularization = regularization,
            lambda_        = lambda_,
        )
        print(f"Model loaded from '{filepath}'.")
        return model

    def summary(self):
        print("=" * 55)
        print(f"{'FFNN Architecture':^55}")
        print("=" * 55)
        total = 0
        for i, layer in enumerate(self.layers):
            params = layer.W.size + layer.b.size
            total += params
            print(f"  Layer {i+1}: {layer}")
            print(f"           params = {params:,}")
        print("-" * 55)
        print(f"  Total trainable parameters: {total:,}")
        print("=" * 55)

    def __repr__(self):
        return f"FFNN(layers={len(self.layers)})"
