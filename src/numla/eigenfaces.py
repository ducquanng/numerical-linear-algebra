"""Face recognition on a hand-written SVD.

Faces live near a low-dimensional subspace of pixel space. The leading singular
vectors of the centred training images span it, and projecting onto them turns a
4,096-pixel image into a few dozen coordinates that a nearest-neighbour rule can
classify.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .svd import power_svd


@dataclass
class EigenfaceModel:
    mean: np.ndarray
    components: np.ndarray      # (pixels, k)
    weights: np.ndarray         # (n_train, k)
    labels: np.ndarray

    def project(self, images: np.ndarray) -> np.ndarray:
        return (images - self.mean) @ self.components

    def predict(self, images: np.ndarray) -> np.ndarray:
        """Nearest training face in the projected space, by Euclidean distance."""
        coordinates = self.project(images)
        distances = ((coordinates[:, None, :] - self.weights[None, :, :]) ** 2).sum(axis=2)
        return self.labels[np.argmin(distances, axis=1)]


def fit(images: np.ndarray, labels: np.ndarray, k: int = 40, seed: int = 0) -> EigenfaceModel:
    """Centre the training set, take its leading right singular vectors as the basis.

    Centring is not cosmetic: without it the first component is the average face
    and every projection is dominated by overall brightness rather than by what
    distinguishes one face from another.
    """
    mean = images.mean(axis=0)
    centred = images - mean
    _, _, V = power_svd(centred, k=k, seed=seed)
    return EigenfaceModel(mean=mean, components=V, weights=centred @ V, labels=labels)


def accuracy(model: EigenfaceModel, images: np.ndarray, labels: np.ndarray) -> float:
    return float((model.predict(images) == labels).mean())
