import numpy as np
import pytest

from numla.eigenfaces import accuracy, fit
from numla.pagerank import pagerank_eigen, pagerank_power, transition_matrix

# Page 2 is a dangling node: it receives links and gives none.
GRAPH = np.array([[0, 1, 0], [1, 0, 0], [1, 1, 0]], dtype=float)


def test_transition_matrix_columns_sum_to_one():
    M = transition_matrix(GRAPH)
    assert np.allclose(M.sum(axis=0), 1.0)


def test_dangling_column_becomes_uniform_instead_of_leaking_probability():
    M = transition_matrix(GRAPH, damping=1.0)
    assert np.allclose(M[:, 2], 1 / 3)


def test_ranks_are_a_probability_distribution():
    ranks, _ = pagerank_power(GRAPH)
    assert ranks.sum() == pytest.approx(1.0)
    assert (ranks > 0).all()


def test_power_iteration_agrees_with_the_eigendecomposition():
    ranks, _ = pagerank_power(GRAPH)
    assert np.allclose(ranks, pagerank_eigen(GRAPH), atol=1e-8)


def test_a_more_linked_page_outranks_a_less_linked_one():
    graph = np.array([[0, 0, 0], [1, 0, 1], [1, 0, 0]], dtype=float)  # page 1 gets two links
    ranks, _ = pagerank_power(graph)
    assert ranks[1] > ranks[0] and ranks[1] > ranks[2]


def test_no_damping_on_a_symmetric_graph_gives_equal_ranks():
    ring = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    ranks, _ = pagerank_power(ring)
    assert np.allclose(ranks, 1 / 3, atol=1e-8)


def synthetic_faces(n_people=6, per_person=8, pixels=64, seed=0):
    """Each identity is one random template plus noise: separable, but not trivially."""
    rng = np.random.default_rng(seed)
    templates = rng.normal(size=(n_people, pixels))
    images, labels = [], []
    for person in range(n_people):
        for _ in range(per_person):
            images.append(templates[person] + rng.normal(scale=0.25, size=pixels))
            labels.append(person)
    return np.array(images), np.array(labels)


def test_eigenfaces_recognises_identities_it_was_trained_on():
    images, labels = synthetic_faces()
    train = np.arange(len(labels)) % 8 < 6
    model = fit(images[train], labels[train], k=10)
    assert accuracy(model, images[~train], labels[~train]) > 0.9


def test_the_basis_is_orthonormal_and_the_mean_is_removed():
    images, labels = synthetic_faces(seed=1)
    model = fit(images, labels, k=5)
    assert np.allclose(model.components.T @ model.components, np.eye(5), atol=1e-6)
    assert np.allclose(model.project(model.mean[None, :]), 0.0, atol=1e-8)


def test_more_components_never_hurt_on_this_easy_problem():
    images, labels = synthetic_faces(seed=2)
    train = np.arange(len(labels)) % 8 < 6
    scores = [accuracy(fit(images[train], labels[train], k=k), images[~train], labels[~train])
              for k in (2, 10)]
    assert scores[1] >= scores[0]
