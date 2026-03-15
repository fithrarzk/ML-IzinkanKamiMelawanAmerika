import numpy as np

from autograd import Tensor
from ffnn import FFNN

def _get_plt():
    """Lazy-import matplotlib so the module can be imported without it."""
    import matplotlib.pyplot as plt
    return plt


def _stats(arr: np.ndarray) -> dict:
    flat = arr.ravel()
    hist_counts, hist_bins = np.histogram(flat, bins=30)
    return {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "hist_counts": hist_counts,
        "hist_bins": hist_bins,
    }


def get_weight_stats(model) -> dict:
    return {i: _stats(layer.W.data) for i, layer in enumerate(model.layers)}


def get_gradient_stats(model) -> dict:
    return {i: _stats(layer.W.grad if layer.W.grad is not None else np.zeros_like(layer.W.data))
            for i, layer in enumerate(model.layers)}


def numerical_gradient(
    model, X: np.ndarray, y: np.ndarray, layer_idx: int, eps: float = 1e-5
) -> np.ndarray:
    """
    Hitung gradien numerik dengan finite differences.
    Digunakan untuk verifikasi bahwa gradien autograd sudah benar.
    """
    layer = model.layers[layer_idx]
    W     = layer.W.data          # akses numpy array di dalam Tensor
    grad  = np.zeros_like(W)

    it = np.nditer(W, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index

        W[idx] += eps
        out_plus = model.forward(X)
        loss_plus = float(model.loss_fn.forward(out_plus, y).data.flat[0])

        W[idx] -= 2 * eps
        out_minus = model.forward(X)
        loss_minus = float(model.loss_fn.forward(out_minus, y).data.flat[0])

        W[idx] += eps           # restore
        grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        it.iternext()

    return grad


def weight_histograms(model) -> list[dict]:
    result = []
    stats = get_weight_stats(model)
    for i, s in stats.items():
        result.append(
            {
                "label": f"Layer {i+1} weights",
                "counts": s["hist_counts"],
                "bins": s["hist_bins"],
            }
        )
    return result


def gradient_histograms(model) -> list[dict]:
    result = []
    stats = get_gradient_stats(model)
    for i, s in stats.items():
        result.append(
            {
                "label": f"Layer {i+1} gradients",
                "counts": s["hist_counts"],
                "bins": s["hist_bins"],
            }
        )
    return result


def loss_curve(history: dict):
    train = np.array(history.get("train_loss", []))
    val = np.array(history["val_loss"]) if history.get("val_loss") else None
    return train, val


def _evaluate_predictions(task, y_prob, y_test):
    if task == "binary":
        y_pred_cls = (y_prob >= 0.5).astype(int).ravel()
        y_true_cls = y_test.astype(int).ravel()
    else:
        y_pred_cls = np.argmax(y_prob, axis=1)
        y_true_cls = np.argmax(y_test, axis=1)
    return float(np.mean(y_pred_cls == y_true_cls))


def run_architecture_experiments(
    experiments,
    task,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    init,
    loss=None,
    lr=0.01,
    regularization="l2",
    lambda_=1e-4,
    epochs=200,
    batch_size=32,
    verbose=20,
):
    results = {}
    loss_name = loss or ("cce" if task == "multiclass" else "bce")

    for exp_name, layer_config in experiments.items():
        print(f"\n{'=' * 60}")
        print(f"Running: {exp_name}")
        print(f"{'=' * 60}")

        config_with_init = [{**layer, **init} for layer in layer_config]

        model = FFNN(config_with_init)
        model.compile(
            loss=loss_name,
            lr=lr,
            regularization=regularization,
            lambda_=lambda_,
        )
        model.summary()

        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            X_val=X_val,
            y_val=y_val,
            verbose=verbose,
        )

        y_prob = model.predict(X_test)
        accuracy = _evaluate_predictions(task, y_prob, y_test)
        test_loss = float(model.loss_fn.forward(y_prob, y_test))

        results[exp_name] = {
            "model": model,
            "history": history,
            "test_loss": test_loss,
            "test_accuracy": accuracy,
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "lr": lr,
        }

        print(f"\n{exp_name} - Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")

    return results


def run_learning_rate_experiments(
    learning_rates,
    base_arch,
    task,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    init,
    loss=None,
    regularization="l2",
    lambda_=1e-4,
    epochs=200,
    batch_size=32,
    verbose=20,
):
    results = {}
    loss_name = loss or ("cce" if task == "multiclass" else "bce")

    for exp_name, lr in learning_rates.items():
        print(f"\n{'=' * 60}")
        print(f"Running: {exp_name} (lr={lr})")
        print(f"{'=' * 60}")

        config_with_init = [{**layer, **init} for layer in base_arch]

        model = FFNN(config_with_init)
        model.compile(
            loss=loss_name,
            lr=lr,
            regularization=regularization,
            lambda_=lambda_,
        )
        model.summary()

        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            X_val=X_val,
            y_val=y_val,
            verbose=verbose,
        )

        y_prob = model.predict(X_test)
        accuracy = _evaluate_predictions(task, y_prob, y_test)
        test_loss = float(model.loss_fn.forward(y_prob, y_test))

        results[exp_name] = {
            "model": model,
            "history": history,
            "test_loss": test_loss,
            "test_accuracy": accuracy,
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "lr": lr,
        }

        print(f"\n{exp_name} - Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")

    return results


def plot_train_val_curves(
    results, title="Training and Validation Loss", zoom_start=None
):
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(8, 5))

    for exp_name, res in results.items():
        ax.plot(res["train_loss"], label=f"{exp_name} (train)", linestyle="-")
        ax.plot(res["val_loss"], label=f"{exp_name} (val)", linestyle="--")

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_prediction_bars(results, title_prefix="Eksperimen"):
    plt = _get_plt()
    exp_names = list(results.keys())
    test_losses = [results[name]["test_loss"] for name in exp_names]
    test_accs = [results[name]["test_accuracy"] for name in exp_names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(exp_names, test_losses, color="steelblue", alpha=0.8)
    axes[0].set_title(f"{title_prefix} - Test Loss")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(exp_names, test_accs, color="seagreen", alpha=0.8)
    axes[1].set_title(f"{title_prefix} - Test Accuracy")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


def extract_layer_weights_and_gradients(model):
    weights_by_layer    = {}
    gradients_by_layer  = {}

    for i, layer in enumerate(model.layers):
        layer_name = f"Layer {i}"
        weights_by_layer[layer_name] = layer.W.data.copy()
        dW = layer.W.grad
        if dW is not None:
            gradients_by_layer[layer_name] = dW.copy()

    return weights_by_layer, gradients_by_layer


def result_analysis(model_triplet):
    print("\n" + "=" * 80)
    print("ANALISIS DISTRIBUSI BOBOT DAN GRADIEN PER LAYER")
    print("=" * 80)

    result_group_name, model_name, result = model_triplet
    model = result["model"]
    weights, gradients = extract_layer_weights_and_gradients(model)

    print(f"\nSumber hasil: {result_group_name}")
    print(f"Model: {model_name}")
    print(f"Test Accuracy: {result['test_accuracy']:.4f}")

    print("\n" + "-" * 80)
    print("STATISTIK BOBOT PER LAYER:")
    print("-" * 80)
    for layer_name, W in weights.items():
        print(f"\n{layer_name}:")
        print(f"  Shape: {W.shape}")
        print(f"  Mean: {np.mean(W):.6f} | Std: {np.std(W):.6f}")
        print(f"  Min: {np.min(W):.6f} | Max: {np.max(W):.6f}")
        print(f"  Median: {np.median(W):.6f}")

    if gradients:
        print("\n" + "-" * 80)
        print("STATISTIK GRADIEN PER LAYER:")
        print("-" * 80)
        for layer_name, dW in gradients.items():
            print(f"\n{layer_name}:")
            print(f"  Shape: {dW.shape}")
            print(f"  Mean: {np.mean(dW):.6f} | Std: {np.std(dW):.6f}")
            print(f"  Min: {np.min(dW):.6f} | Max: {np.max(dW):.6f}")
            print(f"  Median: {np.median(dW):.6f}")
            print(f"  % Zero Gradients: {(np.sum(dW == 0) / dW.size * 100):.2f}%")
    else:
        print("\nGradien belum tersedia.")


def result_vis(weights, gradients):
    plt = _get_plt()
    fig, axes = plt.subplots(len(weights), 2, figsize=(14, 4 * len(weights)))

    if len(weights) == 1:
        axes = axes.reshape(1, -1)

    for idx, (layer_name, W) in enumerate(weights.items()):
        axes[idx, 0].hist(
            W.flatten(), bins=50, edgecolor="black", alpha=0.7, color="steelblue"
        )
        axes[idx, 0].set_title(f"{layer_name} - Distribusi Bobot")
        axes[idx, 0].set_xlabel("Nilai Bobot")
        axes[idx, 0].set_ylabel("Frekuensi")
        axes[idx, 0].grid(True, alpha=0.3)

        if layer_name in gradients:
            dW = gradients[layer_name]
            axes[idx, 1].hist(
                dW.flatten(), bins=50, edgecolor="black", alpha=0.7, color="coral"
            )
            axes[idx, 1].set_title(f"{layer_name} - Distribusi Gradien")
            axes[idx, 1].set_xlabel("Nilai Gradien")
            axes[idx, 1].set_ylabel("Frekuensi")
        else:
            axes[idx, 1].text(
                0.5, 0.5, "Gradien tidak tersedia", ha="center", va="center"
            )
            axes[idx, 1].set_title(f"{layer_name} - Distribusi Gradien")
            axes[idx, 1].set_xticks([])
            axes[idx, 1].set_yticks([])

        axes[idx, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def analyze_results_distributions(results, result_group_name="All Experiments"):
    for exp_name, result in results.items():
        weights, gradients = extract_layer_weights_and_gradients(result["model"])
        result_analysis((result_group_name, exp_name, result))
        result_vis(weights, gradients)
