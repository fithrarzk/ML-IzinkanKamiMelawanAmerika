"""
verify_autograd.py — Verifikasi Gradien Autograd vs Numerical Gradient

Script ini membuktikan bahwa implementasi autograd menghasilkan gradien
yang sama (dalam toleransi numerik) dengan gradien yang dihitung secara
numerik menggunakan finite differences.

Cara kerja finite differences (referensi):
  grad_numerical[i,j] = (L(W[i,j] + eps) - L(W[i,j] - eps)) / (2*eps)

Cara menjalankan:
  cd src
  python verify_autograd.py
"""

import sys
import numpy as np

sys.path.insert(0, ".")
from ffnn import FFNN
from utils import numerical_gradient


def _relative_error(a, b):
    return np.abs(a - b) / (np.abs(a) + np.abs(b) + 1e-15)


def run_check(name, model, X, y, layer_idx=0, eps=1e-5, tol=1e-4):
    """
    Bandingkan gradien autograd (dari loss.backward()) dengan
    gradien numerik (finite differences) untuk layer ke-layer_idx.
    """
    print(f"\n{'='*60}")
    print(f"  Tes: {name}")
    print(f"{'='*60}")

    # --- Autograd gradient ---
    model.zero_grad()
    y_pred   = model.forward(X)
    model.backward(y_pred, y)
    ag_grad  = model.layers[layer_idx].W.grad.copy()

    # --- Numerical gradient ---
    num_grad = numerical_gradient(model, X, y, layer_idx=layer_idx, eps=eps)

    # --- Perbandingan ---
    abs_diff = np.abs(ag_grad - num_grad)
    rel_err  = _relative_error(ag_grad, num_grad)
    max_abs  = float(np.max(abs_diff))
    max_rel  = float(np.max(rel_err))
    mean_abs = float(np.mean(abs_diff))

    print(f"  Autograd grad  [sample]: {ag_grad.ravel()[:5]}")
    print(f"  Numerical grad [sample]: {num_grad.ravel()[:5]}")
    print(f"  Max  |autograd - numerical| = {max_abs:.2e}")
    print(f"  Mean |autograd - numerical| = {mean_abs:.2e}")
    print(f"  Max  relative error        = {max_rel:.2e}")

    passed = max_abs < tol
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  Toleransi = {tol:.0e}  →  {status}")
    return passed


# ──────────────────────────────────────────────────────────
# Model 1: MSE regression (linear output)
# ──────────────────────────────────────────────────────────
np.random.seed(0)
X_reg = np.random.randn(20, 3) * 0.5
y_reg = np.random.randn(20, 1)

model_mse = FFNN([
    {"n_in": 3, "n_out": 4, "activation": "relu",   "init_method": "random_normal", "init_params": {"std": 0.3}},
    {"n_in": 4, "n_out": 1, "activation": "linear", "init_method": "random_normal", "init_params": {"std": 0.3}},
])
model_mse.compile(loss="mse", lr=0.01)

p1 = run_check("MSE + hidden layer (layer 0)", model_mse, X_reg, y_reg, layer_idx=0)
p2 = run_check("MSE + output layer  (layer 1)", model_mse, X_reg, y_reg, layer_idx=1)

# ──────────────────────────────────────────────────────────
# Model 2: BCE binary classification (sigmoid output)
# ──────────────────────────────────────────────────────────
np.random.seed(1)
X_bin = np.random.randn(20, 4) * 0.5
y_bin = (np.random.randn(20, 1) > 0).astype(float)

model_bce = FFNN([
    {"n_in": 4, "n_out": 5,  "activation": "tanh",    "init_method": "random_normal", "init_params": {"std": 0.3}},
    {"n_in": 5, "n_out": 1,  "activation": "sigmoid", "init_method": "random_normal", "init_params": {"std": 0.3}},
])
model_bce.compile(loss="bce", lr=0.01)

p3 = run_check("BCE + hidden layer (layer 0)", model_bce, X_bin, y_bin, layer_idx=0)
p4 = run_check("BCE + output layer  (layer 1)", model_bce, X_bin, y_bin, layer_idx=1)

# ──────────────────────────────────────────────────────────
# Model 3: ReLU deep network — gradient check layer tengah
# ──────────────────────────────────────────────────────────
np.random.seed(2)
X_deep = np.random.randn(15, 3) * 0.3
y_deep = np.random.randn(15, 1)

model_deep = FFNN([
    {"n_in": 3, "n_out": 4, "activation": "relu",   "init_method": "random_normal", "init_params": {"std": 0.2}},
    {"n_in": 4, "n_out": 4, "activation": "relu",   "init_method": "random_normal", "init_params": {"std": 0.2}},
    {"n_in": 4, "n_out": 1, "activation": "linear", "init_method": "random_normal", "init_params": {"std": 0.2}},
])
model_deep.compile(loss="mse", lr=0.01)

p5 = run_check("Deep ReLU (layer 0)", model_deep, X_deep, y_deep, layer_idx=0)
p6 = run_check("Deep ReLU (layer 1)", model_deep, X_deep, y_deep, layer_idx=1)
p7 = run_check("Deep ReLU (layer 2)", model_deep, X_deep, y_deep, layer_idx=2)

# ──────────────────────────────────────────────────────────
# Tes Training — loss harus turun
# ──────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  Tes Training: apakah loss turun setelah 50 epoch?")
print(f"{'='*60}")
np.random.seed(42)
X_t = np.random.randn(100, 4)
y_t = (np.random.randn(100, 1) > 0).astype(float)
model_train = FFNN([
    {"n_in": 4,  "n_out": 8,  "activation": "relu",    "init_method": "random_normal", "init_params": {"std": 0.3}},
    {"n_in": 8,  "n_out": 1,  "activation": "sigmoid", "init_method": "random_normal", "init_params": {"std": 0.3}},
])
model_train.compile(loss="bce", lr=0.05)
h = model_train.fit(X_t, y_t, epochs=100, batch_size=32, verbose=25)
p_train = h["train_loss"][-1] < h["train_loss"][0]
status = "✓ PASS" if p_train else "✗ FAIL"
print(f"  loss: {h['train_loss'][0]:.4f} → {h['train_loss'][-1]:.4f}  {status}")

# ──────────────────────────────────────────────────────────
# Rekap
# ──────────────────────────────────────────────────────────
all_pass = all([p1, p2, p3, p4, p5, p6, p7, p_train])
print(f"\n{'='*60}")
print(f"  HASIL KESELURUHAN: {'✓ SEMUA LULUS' if all_pass else '✗ ADA YANG GAGAL'}")
print(f"{'='*60}\n")
sys.exit(0 if all_pass else 1)
