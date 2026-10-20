"""NumPy replica of the baseline ANN (9-9-1, tansig/purelin, trainlm).

The replica follows the documented configuration of the MATLAB baseline:
one hidden layer of hyperbolic-tangent units, a linear output unit,
Nguyen--Widrow initialisation, Levenberg--Marquardt updates with the toolbox
default damping schedule (mu = 1e-3, x0.1 after success, x10 after failure,
mu_max = 1e10), at most 1000 epochs, minimum gradient 1e-7, and validation
stopping after six consecutive failures, returning the weights of the epoch
with the lowest validation error.
"""
from __future__ import annotations

import numpy as np


class LMNetwork:
    def __init__(
        self,
        n_hidden: int = 9,
        max_epochs: int = 1000,
        mu: float = 1e-3,
        mu_dec: float = 0.1,
        mu_inc: float = 10.0,
        mu_max: float = 1e10,
        max_fail: int = 6,
        min_grad: float = 1e-7,
        seed: int = 0,
    ) -> None:
        self.n_hidden = n_hidden
        self.max_epochs = max_epochs
        self.mu0 = mu
        self.mu_dec = mu_dec
        self.mu_inc = mu_inc
        self.mu_max = mu_max
        self.max_fail = max_fail
        self.min_grad = min_grad
        self.seed = seed

    # -- parameter handling -------------------------------------------------
    def _unpack(self, w: np.ndarray):
        h, p = self.n_hidden, self.n_inputs
        w1 = w[: h * p].reshape(h, p)
        b1 = w[h * p : h * p + h]
        w2 = w[h * p + h : h * p + 2 * h]
        b2 = w[-1]
        return w1, b1, w2, b2

    def _forward(self, w: np.ndarray, x: np.ndarray) -> np.ndarray:
        w1, b1, w2, b2 = self._unpack(w)
        return np.tanh(x @ w1.T + b1) @ w2 + b2

    def _jacobian(self, w: np.ndarray, x: np.ndarray) -> np.ndarray:
        w1, b1, w2, _ = self._unpack(w)
        hidden = np.tanh(x @ w1.T + b1)
        delta = (1.0 - hidden**2) * w2
        n, h, p = x.shape[0], self.n_hidden, self.n_inputs
        jac = np.empty((n, w.size))
        jac[:, : h * p] = (delta[:, :, None] * x[:, None, :]).reshape(n, -1)
        jac[:, h * p : h * p + h] = delta
        jac[:, h * p + h : h * p + 2 * h] = hidden
        jac[:, -1] = 1.0
        return jac

    def _initialise(self, rng: np.random.Generator) -> np.ndarray:
        h, p = self.n_hidden, self.n_inputs
        beta = 0.7 * h ** (1.0 / p)
        w1 = rng.uniform(-1.0, 1.0, (h, p))
        w1 = beta * w1 / np.linalg.norm(w1, axis=1, keepdims=True)
        b1 = beta * np.linspace(-1.0, 1.0, h) * np.sign(rng.uniform(-1.0, 1.0, h))
        w2 = rng.uniform(-1.0, 1.0, h)
        b2 = rng.uniform(-1.0, 1.0)
        return np.concatenate([w1.ravel(), b1, w2, [b2]])

    # -- training -------------------------------------------------------------
    def fit(self, x_train, y_train, x_val, y_val) -> "LMNetwork":
        rng = np.random.default_rng(self.seed)
        self.n_inputs = x_train.shape[1]
        w = self._initialise(rng)
        mu = self.mu0
        error = y_train - self._forward(w, x_train)
        sse = float(error @ error)
        best_val = float(np.mean((y_val - self._forward(w, x_val)) ** 2))
        best_w, best_epoch, fails = w.copy(), 0, 0
        self.epochs_ = 0
        eye = np.eye(w.size)
        for epoch in range(1, self.max_epochs + 1):
            jac = self._jacobian(w, x_train)
            gradient = jac.T @ error
            if 2.0 * np.linalg.norm(gradient) < self.min_grad:
                break
            hessian = jac.T @ jac
            accepted = False
            while mu <= self.mu_max:
                try:
                    step = np.linalg.solve(hessian + mu * eye, gradient)
                except np.linalg.LinAlgError:
                    mu *= self.mu_inc
                    continue
                candidate = w + step
                candidate_error = y_train - self._forward(candidate, x_train)
                candidate_sse = float(candidate_error @ candidate_error)
                if candidate_sse < sse:
                    w, error, sse = candidate, candidate_error, candidate_sse
                    mu = max(mu * self.mu_dec, 1e-20)
                    accepted = True
                    break
                mu *= self.mu_inc
            if not accepted:
                break
            self.epochs_ = epoch
            val = float(np.mean((y_val - self._forward(w, x_val)) ** 2))
            if val < best_val:
                best_val, best_w, best_epoch, fails = val, w.copy(), epoch, 0
            else:
                fails += 1
                if fails >= self.max_fail:
                    break
        self.w_ = best_w
        self.best_epoch_ = best_epoch
        return self

    def predict(self, x) -> np.ndarray:
        return self._forward(self.w_, x)


class MinMaxScaling:
    """Map inputs to [-1, 1] and the target to [0, 1] from a reference set."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MinMaxScaling":
        self.x_min, self.x_max = x.min(axis=0), x.max(axis=0)
        span = self.x_max - self.x_min
        self.x_span = np.where(span > 0, span, 1.0)
        self.y_min, self.y_max = float(y.min()), float(y.max())
        self.y_span = self.y_max - self.y_min if self.y_max > self.y_min else 1.0
        return self

    def x(self, x: np.ndarray) -> np.ndarray:
        return 2.0 * (x - self.x_min) / self.x_span - 1.0

    def y(self, y: np.ndarray) -> np.ndarray:
        return (y - self.y_min) / self.y_span

    def y_inverse(self, y_scaled: np.ndarray) -> np.ndarray:
        return y_scaled * self.y_span + self.y_min


def fit_predict_replica(
    x: np.ndarray,
    y: np.ndarray,
    train_index: np.ndarray,
    predict_index: np.ndarray,
    seed: int,
    scaling_index: np.ndarray | None = None,
    validation_fraction: float = 0.15 / 0.85,
) -> tuple[np.ndarray, int]:
    """Train on `train_index` (with an internal random validation split) and predict.

    `scaling_index` defaults to `train_index`; passing all rows reproduces the
    toolbox behaviour of configuring processing on the complete data set.
    """
    rng = np.random.default_rng(seed)
    scaling_index = train_index if scaling_index is None else scaling_index
    scaler = MinMaxScaling().fit(x[scaling_index], y[scaling_index])
    shuffled = rng.permutation(train_index)
    n_val = int(round(validation_fraction * len(train_index)))
    val_index, fit_index = shuffled[:n_val], shuffled[n_val:]
    xs, ys = scaler.x(x), scaler.y(y)
    network = LMNetwork(seed=seed).fit(xs[fit_index], ys[fit_index], xs[val_index], ys[val_index])
    return scaler.y_inverse(network.predict(xs[predict_index])), network.epochs_
