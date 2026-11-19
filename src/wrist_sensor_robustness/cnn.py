"""
A small 1D CNN, implemented from scratch in NumPy (forward + manual backward).

Why NumPy instead of PyTorch/TensorFlow: this sandbox's network policy blocks
download.pytorch.org (the CPU-only wheel index), and plain `pip install torch`
from PyPI pulls full CUDA dependency packages (multiple GB) - the wrong fit
for a model the project spec explicitly wants small enough to run on a
laptop. A from-scratch implementation has zero heavy dependencies, makes the
"small" in "small 1D CNN" literal (a few thousand parameters), and the
correctness risk of hand-written backprop is addressed directly: see
`tests/test_cnn.py`, which gradient-checks every layer's analytic gradient
against a numerical finite-difference gradient before any result from this
model is trusted.

Architecture:
    Conv1D(stride=2) -> ReLU -> Conv1D(stride=2) -> ReLU
    [-> Conv1D(stride=2) -> ReLU, optional third block]
    -> GlobalAveragePool(time) -> Dense -> softmax

Each layer exposes forward(x) and backward(grad_output), matching the
standard pattern for manual backprop: forward caches what backward needs,
backward returns the gradient w.r.t. its input and accumulates parameter
gradients on `self`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _im2col_1d(x: np.ndarray, kernel_size: int, stride: int) -> np.ndarray:
    """
    x: (N, C_in, L) -> columns: (N, L_out, C_in * kernel_size)

    Each output position's column is the flattened receptive field, so a
    conv becomes one matmul against a (C_out, C_in*kernel_size) weight matrix.
    """
    n, c_in, length = x.shape
    l_out = (length - kernel_size) // stride + 1
    # Byte-stride trick: build an (N, L_out, C_in, kernel_size) view without copying,
    # then reshape (the reshape here does copy, but only once, and only this).
    s_n, s_c, s_l = x.strides
    windows = np.lib.stride_tricks.as_strided(
        x,
        shape=(n, l_out, c_in, kernel_size),
        strides=(s_n, s_l * stride, s_c, s_l),
        writeable=False,
    )
    return windows.reshape(n, l_out, c_in * kernel_size)


def _col2im_1d(dcol: np.ndarray, input_shape: tuple[int, int, int], kernel_size: int, stride: int) -> np.ndarray:
    """Inverse of _im2col_1d's windowing: scatter-add column gradients back to input positions."""
    n, c_in, length = input_shape
    l_out = dcol.shape[1]
    dcol = dcol.reshape(n, l_out, c_in, kernel_size)
    dx = np.zeros(input_shape, dtype=dcol.dtype)
    # l_out and kernel_size are both small (<=200, <=5) for this project's windows,
    # so a plain double loop is simple, correct, and fast enough - no need for a
    # vectorized scatter here.
    for t in range(l_out):
        start = t * stride
        dx[:, :, start : start + kernel_size] += dcol[:, t, :, :]
    return dx


class Conv1D:
    """1D convolution, no padding ('valid'), with its own parameter gradients."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, rng: np.random.Generator):
        # He initialization: appropriate for the ReLU that follows every conv here.
        fan_in = in_channels * kernel_size
        self.weight = rng.normal(0, np.sqrt(2.0 / fan_in), size=(out_channels, in_channels, kernel_size))
        self.bias = np.zeros(out_channels)
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.d_weight: np.ndarray | None = None
        self.d_bias: np.ndarray | None = None
        self._cache: dict = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (N, C_in, L) -> (N, C_out, L_out)"""
        col = _im2col_1d(x, self.kernel_size, self.stride)  # (N, L_out, C_in*k)
        w_flat = self.weight.reshape(self.out_channels, -1)  # (C_out, C_in*k)
        out = col @ w_flat.T + self.bias  # (N, L_out, C_out)
        self._cache = {"col": col, "x_shape": x.shape}
        return out.transpose(0, 2, 1)  # (N, C_out, L_out)

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """d_out: (N, C_out, L_out) -> d_input: (N, C_in, L)"""
        col = self._cache["col"]  # (N, L_out, C_in*k)
        d_out_t = d_out.transpose(0, 2, 1)  # (N, L_out, C_out)

        w_flat = self.weight.reshape(self.out_channels, -1)
        self.d_weight = np.einsum("nlo,nlc->oc", d_out_t, col).reshape(self.weight.shape)
        self.d_bias = d_out_t.sum(axis=(0, 1))

        d_col = d_out_t @ w_flat  # (N, L_out, C_in*k)
        return _col2im_1d(d_col, self._cache["x_shape"], self.kernel_size, self.stride)

    def params_and_grads(self):
        return [(self.weight, self.d_weight), (self.bias, self.d_bias)]


class ReLU:
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = x > 0
        return x * self._mask

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        return d_out * self._mask

    def params_and_grads(self):
        return []


class GlobalAveragePool1D:
    """(N, C, L) -> (N, C). Lets the network handle any window length."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._length = x.shape[2]
        return x.mean(axis=2)

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        n, c = d_out.shape
        return np.repeat(d_out[:, :, None], self._length, axis=2) / self._length

    def params_and_grads(self):
        return []


class Dense:
    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator):
        self.weight = rng.normal(0, np.sqrt(2.0 / in_features), size=(out_features, in_features))
        self.bias = np.zeros(out_features)
        self.d_weight: np.ndarray | None = None
        self.d_bias: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.weight.T + self.bias

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        self.d_weight = d_out.T @ self._x
        self.d_bias = d_out.sum(axis=0)
        return d_out @ self.weight

    def params_and_grads(self):
        return [(self.weight, self.d_weight), (self.bias, self.d_bias)]


def softmax_cross_entropy(logits: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
    """
    logits: (N, n_classes), labels: (N,) int class indices.
    Returns (mean loss, d_logits) - the combined softmax+CE gradient is the
    standard (softmax_probs - one_hot), which is numerically simpler and more
    stable than differentiating softmax and cross-entropy separately.
    """
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    n = logits.shape[0]
    log_likelihood = -np.log(np.clip(probs[np.arange(n), labels], 1e-12, 1.0))
    loss = log_likelihood.mean()

    d_logits = probs.copy()
    d_logits[np.arange(n), labels] -= 1
    d_logits /= n
    return float(loss), d_logits


@dataclass
class SmallCNN1D:
    """
    Fixed small architecture: Conv-ReLU-Conv-ReLU-GlobalAvgPool-Dense-softmax.

    sklearn-ish API (fit/predict/predict_proba) so it drops into the same
    evaluation code as the gradient-boosted baseline.
    """

    n_channels: int
    n_classes: int
    window_length: int
    conv1_filters: int = 8
    conv2_filters: int = 16
    conv3_filters: int = 0  # 0 disables the optional third conv block
    kernel_size: int = 5
    stride: int = 2
    learning_rate: float = 1e-2
    n_epochs: int = 15
    batch_size: int = 64
    seed: int = 42
    classes_: np.ndarray = field(init=False, default=None)
    history_: list[float] = field(init=False, default_factory=list)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.conv1 = Conv1D(self.n_channels, self.conv1_filters, self.kernel_size, self.stride, rng)
        self.relu1 = ReLU()
        self.conv2 = Conv1D(self.conv1_filters, self.conv2_filters, self.kernel_size, self.stride, rng)
        self.relu2 = ReLU()
        self._layers = [self.conv1, self.relu1, self.conv2, self.relu2]
        last = self.conv2_filters
        if self.conv3_filters:
            self.conv3 = Conv1D(self.conv2_filters, self.conv3_filters, self.kernel_size, self.stride, rng)
            self._layers += [self.conv3, ReLU()]
            last = self.conv3_filters
        self.pool = GlobalAveragePool1D()
        self.dense = Dense(last, self.n_classes, rng)
        self._layers += [self.pool, self.dense]
        self._m = {}
        self._v = {}
        self._t = 0

    def n_parameters(self) -> int:
        return sum(p.size for layer in self._layers for p, _ in layer.params_and_grads())

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """x: (N, L, C) -> logits (N, n_classes)"""
        h = x.transpose(0, 2, 1)  # (N, C, L)
        for layer in self._layers:
            h = layer.forward(h)
        return h

    def _backward(self, d_logits: np.ndarray) -> None:
        d = d_logits
        for layer in reversed(self._layers):
            d = layer.backward(d)

    def _adam_step(self) -> None:
        """Adam optimizer over every layer's (param, grad) pairs. beta1=0.9, beta2=0.999 (standard defaults)."""
        self._t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for layer in self._layers:
            for i, (param, grad) in enumerate(layer.params_and_grads()):
                key = (id(layer), i)
                if key not in self._m:
                    self._m[key] = np.zeros_like(param)
                    self._v[key] = np.zeros_like(param)
                self._m[key] = beta1 * self._m[key] + (1 - beta1) * grad
                self._v[key] = beta2 * self._v[key] + (1 - beta2) * (grad**2)
                m_hat = self._m[key] / (1 - beta1**self._t)
                v_hat = self._v[key] / (1 - beta2**self._t)
                param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SmallCNN1D":
        """X: (N, L, C) raw windows. y: (N,) string or int labels."""
        self.classes_ = np.unique(y)
        label_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([label_to_idx[v] for v in y])

        rng = np.random.default_rng(self.seed)
        n = len(X)
        self.history_ = []
        for epoch in range(self.n_epochs):
            order = rng.permutation(n)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n, self.batch_size):
                idx = order[start : start + self.batch_size]
                logits = self._forward(X[idx])
                loss, d_logits = softmax_cross_entropy(logits, y_idx[idx])
                self._backward(d_logits)
                self._adam_step()
                epoch_loss += loss
                n_batches += 1
            self.history_.append(epoch_loss / max(n_batches, 1))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self._forward(X)
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        idx = self.predict_proba(X).argmax(axis=1)
        return self.classes_[idx]
