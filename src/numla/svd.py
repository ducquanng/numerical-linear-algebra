"""SVD built from the power method, and what breaks if you build it the obvious way."""

from __future__ import annotations

import numpy as np

from .eigen import top_eigenpairs


def power_svd(A: np.ndarray, k: int | None = None, tol: float = 1e-11, seed: int = 0):
    """Truncated SVD via the eigendecomposition of the smaller Gram matrix.

    A = U S V^T, where the columns of V are eigenvectors of A^T A and the singular
    values are the square roots of its eigenvalues.

    Forming A^T A squares the condition number, so singular values below roughly
    sqrt(machine epsilon) times the largest are lost - about 1e-8 in double
    precision. That is acceptable when the small directions are noise, as in
    eigenfaces, and unacceptable when they are the answer. A production routine
    (LAPACK's) never forms the Gram matrix for that reason.
    """
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    k = k or min(m, n)
    gram = A.T @ A if n <= m else A @ A.T
    values, vectors = top_eigenpairs(gram, k, tol=tol, seed=seed)
    values = np.clip(values, 0.0, None)
    singular = np.sqrt(values)

    if n <= m:
        V = vectors
        U = np.zeros((m, k))
        for i in range(k):
            U[:, i] = A @ V[:, i] / singular[i] if singular[i] > 1e-13 else 0.0
    else:
        U = vectors
        V = np.zeros((n, k))
        for i in range(k):
            V[:, i] = A.T @ U[:, i] / singular[i] if singular[i] > 1e-13 else 0.0
    return U, singular, V


def reconstruct(U: np.ndarray, singular: np.ndarray, V: np.ndarray, rank: int | None = None):
    """Rebuild the matrix from its leading components: the best rank-r approximation."""
    r = rank or len(singular)
    return U[:, :r] @ np.diag(singular[:r]) @ V[:, :r].T
