"""Measuring how much orthogonality each QR algorithm loses.

The experiment: factor matrices of increasing condition number, then measure
||I - Q^T Q||, the departure from orthogonality. Theory says classical
Gram-Schmidt loses it in proportion to the condition number squared, modified
Gram-Schmidt in proportion to the condition number, and Householder not at all.
"""

from __future__ import annotations

import numpy as np

from .qr import ALGORITHMS


def hilbert(n: int) -> np.ndarray:
    """H[i, j] = 1 / (i + j + 1): famously, catastrophically ill-conditioned."""
    i = np.arange(n)
    return 1.0 / (i[:, None] + i[None, :] + 1.0)


def vandermonde(m: int, n: int) -> np.ndarray:
    """Columns are powers of equally spaced points: nearly parallel, by construction."""
    x = np.linspace(0, 1, m)
    return np.vander(x, n, increasing=True)


def orthogonality_error(Q: np.ndarray) -> float:
    return float(np.linalg.norm(np.eye(Q.shape[1]) - Q.T @ Q, 2))


def reconstruction_error(A: np.ndarray, Q: np.ndarray, R: np.ndarray) -> float:
    return float(np.linalg.norm(A - Q @ R, 2) / np.linalg.norm(A, 2))


def run(sizes=range(2, 15), matrix=hilbert) -> list[dict]:
    """Factor one matrix per size with every algorithm and record both errors."""
    rows = []
    for n in sizes:
        A = matrix(n) if matrix is hilbert else matrix(2 * n, n)
        condition = float(np.linalg.cond(A))
        for name, algorithm in ALGORITHMS.items():
            Q, R = algorithm(A)
            rows.append(
                {
                    "n": n,
                    "condition": condition,
                    "algorithm": name,
                    "orthogonality_error": orthogonality_error(Q),
                    "reconstruction_error": reconstruction_error(A, Q, R),
                }
            )
    return rows
