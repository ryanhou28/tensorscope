"""Core tensor types for Tensorscope."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import uuid

import numpy as np


class TensorKind(Enum):
    """Classification of tensor types for visualization selection."""

    VECTOR = "vector"
    MATRIX = "matrix"
    IMAGE = "image"
    SPARSE_MATRIX = "sparse_matrix"
    POINTCLOUD = "pointcloud"


@dataclass
class TrackedTensor:
    """A tensor with metadata for tracking through operator graphs.

    Attributes:
        data: The underlying numpy array.
        name: Human-readable name for display.
        kind: Classification for visualization selection.
        tags: Set of semantic tags (e.g., 'symmetric', 'psd', 'orthogonal').
        id: Unique identifier for this tensor instance.
    """

    data: np.ndarray
    name: str
    kind: TensorKind
    tags: frozenset[str] = field(default_factory=frozenset)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the underlying data."""
        return self.data.shape

    @property
    def dtype(self) -> str:
        """Return the dtype of the underlying data as a string."""
        return str(self.data.dtype)

    def with_tags(self, *new_tags: str) -> TrackedTensor:
        """Return a new TrackedTensor with additional tags."""
        return TrackedTensor(
            data=self.data,
            name=self.name,
            kind=self.kind,
            tags=self.tags | frozenset(new_tags),
            id=self.id,
        )

    def __repr__(self) -> str:
        tags_str = ", ".join(sorted(self.tags)) if self.tags else "none"
        return (
            f"TrackedTensor(name={self.name!r}, kind={self.kind.value}, "
            f"shape={self.shape}, dtype={self.dtype}, tags=[{tags_str}])"
        )


@dataclass
class TensorSummary:
    """Summary statistics for a tensor, sent to the frontend.

    This is the serializable representation of a tensor for API responses.
    """

    id: str
    name: str
    kind: str  # TensorKind.value for JSON serialization
    tags: list[str]
    shape: tuple[int, ...]
    dtype: str
    stats: dict[str, Any]
    recommended_views: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "tags": self.tags,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "stats": self.stats,
            "recommended_views": self.recommended_views,
        }


def compute_summary(tensor: TrackedTensor) -> TensorSummary:
    """Compute summary statistics for a tensor.

    Args:
        tensor: The tracked tensor to summarize.

    Returns:
        A TensorSummary with computed statistics.
    """
    data = tensor.data
    stats: dict[str, Any] = {}

    # Basic statistics
    stats["min"] = float(np.min(data))
    stats["max"] = float(np.max(data))
    stats["mean"] = float(np.mean(data))
    stats["std"] = float(np.std(data))

    # Norm (Frobenius for matrices, L2 for vectors)
    stats["norm"] = float(np.linalg.norm(data))

    # Size information
    stats["size"] = int(data.size)
    stats["ndim"] = int(data.ndim)

    # Matrix-specific statistics
    if tensor.kind == TensorKind.MATRIX and data.ndim == 2:
        m, n = data.shape

        # Rank (for reasonable-sized matrices)
        if max(m, n) <= 1000:
            stats["rank"] = int(np.linalg.matrix_rank(data))

        # Condition number (for non-zero matrices)
        if stats["norm"] > 1e-10:
            try:
                stats["condition_number"] = float(np.linalg.cond(data))
            except np.linalg.LinAlgError:
                stats["condition_number"] = float("inf")

        # Check symmetry (for square matrices)
        if m == n:
            stats["is_symmetric"] = bool(np.allclose(data, data.T))

            # Check positive definiteness (for symmetric matrices)
            if stats.get("is_symmetric"):
                try:
                    eigvals = np.linalg.eigvalsh(data)
                    stats["is_positive_definite"] = bool(np.all(eigvals > 0))
                    stats["is_positive_semidefinite"] = bool(np.all(eigvals >= -1e-10))
                    stats["min_eigenvalue"] = float(np.min(eigvals))
                    stats["max_eigenvalue"] = float(np.max(eigvals))
                except np.linalg.LinAlgError:
                    pass

        # Singular values (for reasonable-sized matrices)
        if max(m, n) <= 500:
            try:
                singular_values = np.linalg.svd(data, compute_uv=False)
                stats["singular_values"] = [float(s) for s in singular_values]
                stats["max_singular_value"] = float(singular_values[0])
                stats["min_singular_value"] = float(singular_values[-1])
            except np.linalg.LinAlgError:
                pass

    # Sparsity
    zero_count = np.sum(np.abs(data) < 1e-10)
    stats["sparsity"] = float(zero_count / data.size)

    # Determine recommended views based on kind and tags
    recommended_views = _get_recommended_views(tensor, stats)

    return TensorSummary(
        id=tensor.id,
        name=tensor.name,
        kind=tensor.kind.value,
        tags=sorted(tensor.tags),
        shape=tensor.shape,
        dtype=tensor.dtype,
        stats=stats,
        recommended_views=recommended_views,
    )


def _get_recommended_views(tensor: TrackedTensor, stats: dict[str, Any]) -> list[str]:
    """Determine recommended visualization types for a tensor."""
    views: list[str] = []

    if tensor.kind == TensorKind.VECTOR:
        views.append("vector_stem")
        if len(tensor.data) <= 50:
            views.append("bar_chart")

    elif tensor.kind == TensorKind.MATRIX:
        views.append("heatmap")

        # For 2x2 matrices, ellipse visualization is useful
        if tensor.shape == (2, 2):
            views.append("ellipse_2d")

        # If we have singular values, recommend that view
        if "singular_values" in stats:
            views.append("singular_values")

        # For symmetric PSD matrices, eigenvalue view
        if stats.get("is_symmetric") and stats.get("is_positive_semidefinite"):
            views.append("eigenvalues")

    elif tensor.kind == TensorKind.IMAGE:
        views.append("image")

    elif tensor.kind == TensorKind.SPARSE_MATRIX:
        views.append("sparsity_pattern")
        views.append("heatmap")

    elif tensor.kind == TensorKind.POINTCLOUD:
        if tensor.data.shape[-1] == 2:
            views.append("scatter_2d")
        elif tensor.data.shape[-1] == 3:
            views.append("scatter_3d")

    return views
