import numpy as np
import pytest

from numla.qr import ALGORITHMS, givens_rotation, householder_qr, modified_gram_schmidt
from numla.stability import hilbert

SHAPES = [(3, 3), (6, 4), (40, 12), (5, 5)]


def random_matrix(shape, seed=0):
    return np.random.default_rng(seed).normal(size=shape)


@pytest.mark.parametrize("name", list(ALGORITHMS))
@pytest.mark.parametrize("shape", SHAPES)
def test_factorisation_reconstructs_the_matrix(name, shape):
    A = random_matrix(shape)
    Q, R = ALGORITHMS[name](A)
    assert np.allclose(Q @ R, A, atol=1e-10)


@pytest.mark.parametrize("name", list(ALGORITHMS))
@pytest.mark.parametrize("shape", SHAPES)
def test_q_is_orthonormal_and_r_is_upper_triangular(name, shape):
    A = random_matrix(shape, seed=1)
    Q, R = ALGORITHMS[name](A)
    assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-10)
    assert np.allclose(np.tril(R, -1), 0, atol=1e-10)


@pytest.mark.parametrize("name", list(ALGORITHMS))
def test_agrees_with_lapack_up_to_column_signs(name):
    """QR is unique only up to the sign of each column, so compare |R| diagonals."""
    A = random_matrix((8, 5), seed=2)
    _, R = ALGORITHMS[name](A)
    _, reference = np.linalg.qr(A)
    assert np.allclose(np.abs(np.diag(R)), np.abs(np.diag(reference)), atol=1e-10)


def test_givens_rotation_zeroes_the_second_entry():
    for a, b in [(3.0, 4.0), (-1.0, 2.0), (5.0, 0.0), (1e300, 1e300)]:
        c, s = givens_rotation(a, b)
        rotated = np.array([[c, -s], [s, c]]) @ np.array([a, b])
        assert abs(rotated[1]) < 1e-10 * max(1.0, abs(a), abs(b))
        assert c**2 + s**2 == pytest.approx(1.0)


def test_givens_does_not_overflow_on_huge_entries():
    """np.hypot exists for this: sqrt(a*a + b*b) overflows here and returns nan."""
    c, s = givens_rotation(1e200, 1e200)
    assert np.isfinite(c) and np.isfinite(s)


def test_householder_beats_classical_gram_schmidt_when_it_matters():
    """The headline result of the study, asserted rather than only plotted."""
    A = hilbert(10)
    orthogonality = {}
    for name in ("Classical Gram-Schmidt", "Modified Gram-Schmidt", "Householder"):
        Q, _ = ALGORITHMS[name](A)
        orthogonality[name] = np.linalg.norm(np.eye(Q.shape[1]) - Q.T @ Q, 2)
    assert orthogonality["Householder"] < 1e-12
    assert orthogonality["Modified Gram-Schmidt"] < orthogonality["Classical Gram-Schmidt"]
    assert orthogonality["Classical Gram-Schmidt"] > 0.5  # no longer orthogonal at all


def test_modified_gram_schmidt_handles_a_rank_deficient_matrix_without_crashing():
    A = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])  # second column is twice the first
    Q, R = modified_gram_schmidt(A)
    assert np.allclose(Q @ R, A, atol=1e-10)
    assert abs(R[1, 1]) < 1e-12


def test_householder_on_a_tall_thin_matrix_returns_reduced_shapes():
    Q, R = householder_qr(random_matrix((30, 4), seed=3))
    assert Q.shape == (30, 4)
    assert R.shape == (4, 4)
