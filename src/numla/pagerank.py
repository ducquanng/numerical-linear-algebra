"""PageRank three ways, on the same matrix.

A page's rank is the probability that a random surfer is on it. That makes the
rank vector the dominant eigenvector of the link matrix - so the power method is
not an approximation to PageRank, it *is* PageRank as Google first described it.
"""

from __future__ import annotations

import numpy as np

from .eigen import power_method


def transition_matrix(adjacency: np.ndarray, damping: float = 0.85) -> np.ndarray:
    """Column-stochastic Google matrix: M = d * S + (1 - d) / n.

    Two corrections make the plain link matrix usable. A page with no outgoing
    links (a dangling node) would leak probability, so its column is replaced by a
    uniform one. The damping term then mixes in a small chance of jumping to a
    random page, which makes the chain irreducible and guarantees a unique
    stationary distribution - without it a disconnected component can trap the
    surfer and the eigenvector is not unique.
    """
    A = np.asarray(adjacency, dtype=float)
    n = A.shape[0]
    column_sums = A.sum(axis=0)
    S = np.divide(A, column_sums, out=np.full_like(A, 1.0 / n), where=column_sums != 0)
    return damping * S + (1 - damping) / n


def pagerank_power(adjacency: np.ndarray, damping: float = 0.85, tol: float = 1e-12):
    """Iterate the Google matrix to its stationary distribution."""
    M = transition_matrix(adjacency, damping)
    result = power_method(M, tol=tol, x0=np.ones(M.shape[0]) / M.shape[0])
    ranks = np.abs(result.vector)
    return ranks / ranks.sum(), result


def pagerank_eigen(adjacency: np.ndarray, damping: float = 0.85) -> np.ndarray:
    """Same answer from a full eigendecomposition, as a cross-check.

    Exact and unusable at web scale: this is a dense O(n^3) factorisation of a
    matrix with billions of rows. The power method needs only matrix-vector
    products against a sparse matrix, which is the entire reason it is used.
    """
    M = transition_matrix(adjacency, damping)
    values, vectors = np.linalg.eig(M)
    dominant = np.abs(vectors[:, np.argmin(np.abs(values - 1.0))].real)
    return dominant / dominant.sum()


def sensitivity_to_damping(adjacency: np.ndarray, dampings) -> dict[float, np.ndarray]:
    return {d: pagerank_power(adjacency, damping=d)[0] for d in dampings}
