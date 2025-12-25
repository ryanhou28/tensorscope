"""Matrix decomposition operators for Tensorscope."""

from __future__ import annotations

import numpy as np

from ..core.operator import Operator, TensorSpec
from ..core.tensor import TrackedTensor, TensorKind


class SVD(Operator):
    """Singular Value Decomposition operator.

    Computes A = U @ diag(S) @ Vt where:
        - U: Left singular vectors (m x k)
        - S: Singular values (k,) where k = min(m, n)
        - Vt: Right singular vectors transposed (k x n)
    """

    def __init__(self, full_matrices: bool = False):
        """Initialize SVD operator.

        Args:
            full_matrices: If True, U and Vt are full unitary matrices.
                          If False (default), only the first k columns/rows.
        """
        self._full_matrices = full_matrices

    @property
    def name(self) -> str:
        return "SVD"

    @property
    def description(self) -> str:
        return "Singular Value Decomposition: A = U @ diag(S) @ Vt"

    @property
    def tags(self) -> frozenset[str]:
        return frozenset({"deterministic", "decomposition"})

    @property
    def input_specs(self) -> dict[str, TensorSpec]:
        return {
            "A": TensorSpec(
                name="A",
                kind=TensorKind.MATRIX,
                shape=(-1, -1),
                description="Input matrix to decompose",
            ),
        }

    @property
    def output_specs(self) -> dict[str, TensorSpec]:
        return {
            "U": TensorSpec(
                name="U",
                kind=TensorKind.MATRIX,
                description="Left singular vectors",
            ),
            "S": TensorSpec(
                name="S",
                kind=TensorKind.VECTOR,
                description="Singular values",
            ),
            "Vt": TensorSpec(
                name="Vt",
                kind=TensorKind.MATRIX,
                description="Right singular vectors (transposed)",
            ),
        }

    def forward(
        self, inputs: dict[str, TrackedTensor]
    ) -> dict[str, TrackedTensor]:
        A = inputs["A"]

        U, S, Vt = np.linalg.svd(A.data, full_matrices=self._full_matrices)

        return {
            "U": TrackedTensor(
                data=U,
                name=f"U({A.name})",
                kind=TensorKind.MATRIX,
                tags=frozenset({"orthogonal", "left_singular_vectors"}),
            ),
            "S": TrackedTensor(
                data=S,
                name=f"σ({A.name})",
                kind=TensorKind.VECTOR,
                tags=frozenset({"singular_values", "non_negative", "sorted_descending"}),
            ),
            "Vt": TrackedTensor(
                data=Vt,
                name=f"Vt({A.name})",
                kind=TensorKind.MATRIX,
                tags=frozenset({"orthogonal", "right_singular_vectors"}),
            ),
        }


class Eigendecomposition(Operator):
    """Eigendecomposition operator for symmetric/Hermitian matrices.

    Computes A = V @ diag(λ) @ V^T where:
        - V: Eigenvector matrix (columns are eigenvectors)
        - λ: Eigenvalues (sorted ascending)

    Note: Uses np.linalg.eigh which assumes the input is symmetric.
    For general matrices, use a different operator.
    """

    @property
    def name(self) -> str:
        return "Eigendecomposition"

    @property
    def description(self) -> str:
        return "Eigendecomposition for symmetric matrices: A = V @ diag(λ) @ V^T"

    @property
    def tags(self) -> frozenset[str]:
        return frozenset({"deterministic", "decomposition"})

    @property
    def input_specs(self) -> dict[str, TensorSpec]:
        return {
            "A": TensorSpec(
                name="A",
                kind=TensorKind.MATRIX,
                shape=(-1, -1),
                description="Symmetric input matrix",
            ),
        }

    @property
    def output_specs(self) -> dict[str, TensorSpec]:
        return {
            "eigenvalues": TensorSpec(
                name="eigenvalues",
                kind=TensorKind.VECTOR,
                description="Eigenvalues (ascending order)",
            ),
            "eigenvectors": TensorSpec(
                name="eigenvectors",
                kind=TensorKind.MATRIX,
                description="Eigenvector matrix (columns are eigenvectors)",
            ),
        }

    def forward(
        self, inputs: dict[str, TrackedTensor]
    ) -> dict[str, TrackedTensor]:
        A = inputs["A"]

        # Use eigh for symmetric matrices (guaranteed real eigenvalues)
        eigenvalues, eigenvectors = np.linalg.eigh(A.data)

        # Determine tags for eigenvalues
        eig_tags: set[str] = {"eigenvalues", "sorted_ascending"}
        if np.all(eigenvalues > 0):
            eig_tags.add("all_positive")
        elif np.all(eigenvalues >= -1e-10):
            eig_tags.add("all_non_negative")
        elif np.all(eigenvalues < 0):
            eig_tags.add("all_negative")

        return {
            "eigenvalues": TrackedTensor(
                data=eigenvalues,
                name=f"λ({A.name})",
                kind=TensorKind.VECTOR,
                tags=frozenset(eig_tags),
            ),
            "eigenvectors": TrackedTensor(
                data=eigenvectors,
                name=f"V({A.name})",
                kind=TensorKind.MATRIX,
                tags=frozenset({"orthogonal", "eigenvectors"}),
            ),
        }


class QR(Operator):
    """QR decomposition operator.

    Computes A = Q @ R where:
        - Q: Orthogonal matrix (m x k)
        - R: Upper triangular matrix (k x n)
        - k = min(m, n)
    """

    def __init__(self, mode: str = "reduced"):
        """Initialize QR operator.

        Args:
            mode: 'reduced' (default) or 'complete'.
                  'reduced' returns Q: (m, k) and R: (k, n)
                  'complete' returns Q: (m, m) and R: (m, n)
        """
        self._mode = mode

    @property
    def name(self) -> str:
        return "QR"

    @property
    def description(self) -> str:
        return "QR decomposition: A = Q @ R"

    @property
    def tags(self) -> frozenset[str]:
        return frozenset({"deterministic", "decomposition"})

    @property
    def input_specs(self) -> dict[str, TensorSpec]:
        return {
            "A": TensorSpec(
                name="A",
                kind=TensorKind.MATRIX,
                shape=(-1, -1),
                description="Input matrix",
            ),
        }

    @property
    def output_specs(self) -> dict[str, TensorSpec]:
        return {
            "Q": TensorSpec(
                name="Q",
                kind=TensorKind.MATRIX,
                description="Orthogonal matrix",
            ),
            "R": TensorSpec(
                name="R",
                kind=TensorKind.MATRIX,
                description="Upper triangular matrix",
            ),
        }

    def forward(
        self, inputs: dict[str, TrackedTensor]
    ) -> dict[str, TrackedTensor]:
        A = inputs["A"]

        Q, R = np.linalg.qr(A.data, mode=self._mode)

        return {
            "Q": TrackedTensor(
                data=Q,
                name=f"Q({A.name})",
                kind=TensorKind.MATRIX,
                tags=frozenset({"orthogonal"}),
            ),
            "R": TrackedTensor(
                data=R,
                name=f"R({A.name})",
                kind=TensorKind.MATRIX,
                tags=frozenset({"upper_triangular"}),
            ),
        }


class Cholesky(Operator):
    """Cholesky decomposition operator.

    Computes A = L @ L^T where L is lower triangular.
    Only valid for positive definite matrices.
    """

    @property
    def name(self) -> str:
        return "Cholesky"

    @property
    def description(self) -> str:
        return "Cholesky decomposition: A = L @ L^T (for positive definite A)"

    @property
    def tags(self) -> frozenset[str]:
        return frozenset({"deterministic", "decomposition"})

    @property
    def input_specs(self) -> dict[str, TensorSpec]:
        return {
            "A": TensorSpec(
                name="A",
                kind=TensorKind.MATRIX,
                shape=(-1, -1),
                description="Positive definite input matrix",
            ),
        }

    @property
    def output_specs(self) -> dict[str, TensorSpec]:
        return {
            "L": TensorSpec(
                name="L",
                kind=TensorKind.MATRIX,
                description="Lower triangular Cholesky factor",
            ),
        }

    def forward(
        self, inputs: dict[str, TrackedTensor]
    ) -> dict[str, TrackedTensor]:
        A = inputs["A"]

        L = np.linalg.cholesky(A.data)

        return {
            "L": TrackedTensor(
                data=L,
                name=f"chol({A.name})",
                kind=TensorKind.MATRIX,
                tags=frozenset({"lower_triangular", "cholesky_factor"}),
            ),
        }
