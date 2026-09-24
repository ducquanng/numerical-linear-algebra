"""Eigenvalues by iteration: the power method and what it costs.

The power method repeatedly multiplies a vector by A and renormalises. Every
component except the dominant eigenvector shrinks by |lambda_i / lambda_1| per
step, so the error falls geometrically at rate |lambda_2 / lambda_1|. When the
top two eigenvalues are close, that ratio approaches one and convergence stalls -
which is measurable, and `study.py` measures it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EigenResult:
    value: float
    vector: np.ndarray
    iterations: int
    converged: bool
    residuals: list[float] = field(default_factory=list)

    @property
    def observed_rate(self) -> float:
        """Geometric decay of the residual over the last few steps.

        Compared against |lambda_2 / lambda_1| this says whether the iteration is
        behaving as the theory predicts or has stagnated on rounding error.
        """
        useful = [r for r in self.residuals if r > 1e-14]
        if len(useful) < 4:
            return float("nan")
        ratios = [useful[i + 1] / useful[i] for i in range(len(useful) - 4, len(useful) - 1)]
        return float(np.mean(ratios))


def power_method(A: np.ndarray, tol: float = 1e-12, max_iter: int = 5000,
                 seed: int = 0, x0: np.ndarray | None = None) -> EigenResult:
    """Dominant eigenpair by repeated multiplication.

    The eigenvalue is taken as the Rayleigh quotient rather than a ratio of
    entries: it is the least-squares best scalar for the current vector, and its
    error is the square of the vector's error for a symmetric matrix.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    x = np.asarray(x0, dtype=float) if x0 is not None else np.random.default_rng(seed).normal(size=n)
    x = x / np.linalg.norm(x)

    residuals: list[float] = []
    value = float(x @ A @ x)
    for iteration in range(1, max_iter + 1):
        y = A @ x
        norm = np.linalg.norm(y)
        if norm == 0:
            return EigenResult(0.0, x, iteration, True, residuals)
        x = y / norm
        value = float(x @ A @ x)
        residual = float(np.linalg.norm(A @ x - value * x))
        residuals.append(residual)
        if residual < tol:
            return EigenResult(value, x, iteration, True, residuals)
    return EigenResult(value, x, max_iter, False, residuals)


def deflate(A: np.ndarray, value: float, vector: np.ndarray) -> np.ndarray:
    """Remove a known eigenpair so the next power iteration finds the next one.

    Valid for a symmetric matrix, where eigenvectors are orthogonal. Deflation
    accumulates error: each removed pair carries its own into what remains, which
    is why libraries do not build a full spectrum this way.
    """
    v = vector / np.linalg.norm(vector)
    return A - value * np.outer(v, v)


def top_eigenpairs(A: np.ndarray, k: int, tol: float = 1e-11, seed: int = 0):
    """The k dominant eigenpairs of a symmetric matrix, by power method plus deflation."""
    working = np.asarray(A, dtype=float).copy()
    values, vectors = [], []
    for i in range(k):
        result = power_method(working, tol=tol, seed=seed + i)
        values.append(result.value)
        vectors.append(result.vector)
        working = deflate(working, result.value, result.vector)
    return np.array(values), np.column_stack(vectors)
