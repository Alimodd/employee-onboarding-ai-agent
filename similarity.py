"""Small, self-contained cosine-similarity utility for the embedding demo.

Kept separate from the embedding script so it can be unit-tested without any
API calls. Uses NumPy for clear, vectorised math.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Return the cosine similarity between two equal-length numeric vectors.

    Cosine similarity measures the angle between vectors, ignoring magnitude:
        cos = (a . b) / (||a|| * ||b||)

    Args:
        vec_a: First vector.
        vec_b: Second vector (must have the same dimension as ``vec_a``).

    Returns:
        A float in roughly [-1, 1]. Returns 0.0 if either vector has zero length
        (so a zero vector never crashes with a divide-by-zero).

    Raises:
        ValueError: If the two vectors have different dimensions or are empty.
    """
    a = np.asarray(vec_a, dtype=float)
    b = np.asarray(vec_b, dtype=float)

    if a.size == 0 or b.size == 0:
        raise ValueError("cosine_similarity received an empty vector.")
    if a.shape != b.shape:
        raise ValueError(
            f"Vector dimension mismatch: {a.shape} vs {b.shape}."
        )

    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        # A zero-length vector has no direction; define similarity as 0.0.
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))
