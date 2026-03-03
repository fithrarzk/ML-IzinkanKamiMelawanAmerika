import numpy as np


def _stats(arr: np.ndarray) -> dict:
    flat = arr.ravel()
    hist_counts, hist_bins = np.histogram(flat, bins=30)
    return {
        'mean':       float(np.mean(flat)),
        'std':        float(np.std(flat)),
        'min':        float(np.min(flat)),
        'max':        float(np.max(flat)),
        'hist_counts': hist_counts,
        'hist_bins':   hist_bins,
    }


def get_weight_stats(model) -> dict:
    return {i: _stats(layer.W) for i, layer in enumerate(model.layers)}


def get_gradient_stats(model) -> dict:
    return {i: _stats(layer.dW) for i, layer in enumerate(model.layers)}

def numerical_gradient(model, X: np.ndarray, y: np.ndarray,
                        layer_idx: int, eps: float = 1e-5) -> np.ndarray:
    layer = model.layers[layer_idx]
    W     = layer.W
    grad  = np.zeros_like(W)

    it = np.nditer(W, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index

        W[idx] += eps
        loss_plus = model.loss_fn.forward(model.forward(X), y)

        W[idx] -= 2 * eps
        loss_minus = model.loss_fn.forward(model.forward(X), y)

        W[idx] += eps                          # restore original value
        grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        it.iternext()

    return grad

def weight_histograms(model) -> list[dict]:
    result = []
    stats  = get_weight_stats(model)
    for i, s in stats.items():
        result.append({
            'label':  f"Layer {i+1} weights",
            'counts': s['hist_counts'],
            'bins':   s['hist_bins'],
        })
    return result


def gradient_histograms(model) -> list[dict]:
    result = []
    stats  = get_gradient_stats(model)
    for i, s in stats.items():
        result.append({
            'label':  f"Layer {i+1} gradients",
            'counts': s['hist_counts'],
            'bins':   s['hist_bins'],
        })
    return result


def loss_curve(history: dict):
    train = np.array(history.get('train_loss', []))
    val   = np.array(history['val_loss']) if history.get('val_loss') else None
    return train, val
