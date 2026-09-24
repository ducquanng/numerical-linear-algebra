import numpy as np
import pytest

from numla.eigen import deflate, power_method, top_eigenpairs
from numla.svd import power_svd, reconstruct


def symmetric(n=8, seed=0):
    B = np.random.default_rng(seed).normal(size=(n, n))
    return B @ B.T


def test_power_method_finds_the_dominant_eigenvalue():
    A = symmetric()
    result = power_method(A)
    assert result.converged
    assert result.value == pytest.approx(np.linalg.eigvalsh(A).max(), abs=1e-9)


def test_eigenvector_satisfies_its_defining_equation():
    A = symmetric(seed=1)
    result = power_method(A)
    assert np.allclose(A @ result.vector, result.value * result.vector, atol=1e-8)


def test_convergence_rate_matches_the_eigenvalue_ratio():
    """The theory says the residual decays at |lambda_2 / lambda_1| per step."""
    Q, _ = np.linalg.qr(np.random.default_rng(2).normal(size=(12, 12)))
    spectrum = np.array([1.0, 0.6] + [0.1] * 10)
    A = Q @ np.diag(spectrum) @ Q.T
    result = power_method(A, tol=1e-13)
    assert result.observed_rate == pytest.approx(0.6, abs=0.02)


def test_a_smaller_gap_costs_more_iterations():
    Q, _ = np.linalg.qr(np.random.default_rng(3).normal(size=(10, 10)))
    def iterations(ratio):
        A = Q @ np.diag([1.0, ratio] + [0.05] * 8) @ Q.T
        return power_method(A, tol=1e-12, max_iter=50000).iterations
    assert iterations(0.9) > 3 * iterations(0.3)


def test_deflation_removes_the_pair_it_was_given():
    A = symmetric(seed=4)
    result = power_method(A)
    deflated = deflate(A, result.value, result.vector)
    assert np.linalg.eigvalsh(deflated).max() < result.value - 1e-6


def test_top_eigenpairs_match_lapack():
    A = symmetric(n=10, seed=5)
    values, vectors = top_eigenpairs(A, 4)
    assert np.allclose(values, np.sort(np.linalg.eigvalsh(A))[::-1][:4], atol=1e-7)
    assert np.allclose(vectors.T @ vectors, np.eye(4), atol=1e-6)


def test_singular_values_match_lapack():
    A = np.random.default_rng(6).normal(size=(15, 6))
    _, singular, _ = power_svd(A)
    assert np.allclose(np.sort(singular)[::-1], np.linalg.svd(A, compute_uv=False), atol=1e-8)


def test_svd_reconstructs_the_matrix():
    A = np.random.default_rng(7).normal(size=(12, 5))
    U, s, V = power_svd(A)
    assert np.allclose(reconstruct(U, s, V), A, atol=1e-8)


def test_truncated_svd_is_the_best_low_rank_approximation():
    """Eckart-Young: the error of the rank-r truncation is the next singular value."""
    A = np.random.default_rng(8).normal(size=(14, 6))
    U, s, V = power_svd(A)
    for rank in (1, 3, 5):
        error = np.linalg.norm(A - reconstruct(U, s, V, rank=rank), 2)
        assert error == pytest.approx(np.sort(s)[::-1][rank], abs=1e-6)


def test_svd_works_on_a_wide_matrix_too():
    A = np.random.default_rng(9).normal(size=(5, 14))
    U, s, V = power_svd(A)
    assert np.allclose(reconstruct(U, s, V), A, atol=1e-8)
