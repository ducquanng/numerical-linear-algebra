"""Four QR factorisations, and why a textbook one should not be used.

Every routine returns (Q, R) with A = QR and Q orthonormal. They differ only in
how the orthogonalisation is arranged, and that arrangement decides how much
orthogonality survives finite-precision arithmetic:

    classical Gram-Schmidt   loses orthogonality at a rate of about kappa^2
    modified Gram-Schmidt    loses it at about kappa
    Householder / Givens     backward stable; orthogonality stays near machine epsilon

`stability.py` measures exactly that, and the difference is visible by n = 12.
"""

from __future__ import annotations

import numpy as np


def classical_gram_schmidt(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Subtract every earlier projection from the *original* column.

    The projections are all computed against the original vector, so rounding
    error in each one is never corrected by the others. It is the version most
    courses teach first and the one no library uses.
    """
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            R[i, j] = Q[:, i] @ A[:, j]
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        if R[j, j] > 0:
            Q[:, j] = v / R[j, j]
    return Q, R


def modified_gram_schmidt(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Subtract each projection from the *running* vector.

    One line differs from the classical version - the projection is taken against
    `v` rather than `A[:, j]` - and it changes the error growth from kappa squared
    to kappa, because each step orthogonalises what is actually left.
    """
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    V = A.copy()
    for j in range(n):
        R[j, j] = np.linalg.norm(V[:, j])
        if R[j, j] > 0:
            Q[:, j] = V[:, j] / R[j, j]
        for k in range(j + 1, n):
            R[j, k] = Q[:, j] @ V[:, k]
            V[:, k] = V[:, k] - R[j, k] * Q[:, j]
    return Q, R


def _householder_vector(x: np.ndarray) -> np.ndarray:
    """The vector v defining the reflection that maps x onto a multiple of e1."""
    v = x.copy().astype(float)
    alpha = -np.copysign(np.linalg.norm(x), x[0]) if x[0] != 0 else -np.linalg.norm(x)
    v[0] -= alpha
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def householder_qr(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reflect one column at a time into upper-triangular form.

    Each step applies an orthogonal matrix exactly rather than building Q from
    accumulated subtractions, which is why the result stays orthogonal.
    """
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    R = A.copy()
    Q = np.eye(m)
    for j in range(min(m - 1, n)):
        v = _householder_vector(R[j:, j])
        R[j:, j:] -= 2.0 * np.outer(v, v @ R[j:, j:])
        Q[:, j:] -= 2.0 * np.outer(Q[:, j:] @ v, v)
    return Q[:, :n], R[:n, :]


def givens_rotation(a: float, b: float) -> tuple[float, float]:
    """Cosine and sine of the plane rotation sending (a, b) to (hypot(a, b), 0).

    With M = [[c, -s], [s, c]], M @ (a, b) has second entry c*b + s*a = 0.
    `np.hypot` rather than sqrt(a*a + b*b): the latter overflows for large
    entries and underflows for small ones, which is the whole reason Givens
    rotations are written this way in libraries.
    """
    if b == 0.0:
        return 1.0, 0.0
    r = np.hypot(a, b)
    return a / r, -b / r


def givens_qr(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zero one entry at a time with plane rotations.

    Slower than Householder on a dense matrix, and the method of choice when the
    matrix is sparse or nearly triangular already, because each rotation touches
    only two rows.
    """
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    R = A.copy()
    Q = np.eye(m)
    for j in range(n):
        for i in range(m - 1, j, -1):
            if R[i, j] == 0.0:
                continue
            c, s = givens_rotation(R[i - 1, j], R[i, j])
            M = np.array([[c, -s], [s, c]])
            # R <- M R zeroes the entry; A = QR then needs Q <- Q M^T.
            R[[i - 1, i], j:] = M @ R[[i - 1, i], j:]
            Q[:, [i - 1, i]] = Q[:, [i - 1, i]] @ M.T
    return Q[:, :n], R[:n, :]


ALGORITHMS = {
    "Classical Gram-Schmidt": classical_gram_schmidt,
    "Modified Gram-Schmidt": modified_gram_schmidt,
    "Householder": householder_qr,
    "Givens": givens_qr,
}
