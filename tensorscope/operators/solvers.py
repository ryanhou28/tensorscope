"""Linear system solver operators for Tensorscope."""

from __future__ import annotations

import numpy as np

from ..core.operator import Operator, TensorSpec
from ..core.tensor import TrackedTensor, TensorKind


class LeastSquares(Operator):
    """Least squares solver operator.

    Solves the least squares problem: min_x ||Ax - b||_2

    For overdetermined systems (m > n), this finds the solution that
    minimizes the residual norm. For underdetermined systems (m < n),
    this finds the minimum norm solution.
    """

    @property
    def name(self) -> str:
        return "LeastSquares"

    @property
    def description(self) -> str:
        return "Least squares solver: x = argmin ||Ax - b||"

    @property
    def tags(self) -> frozenset[str]:
        return frozenset({"deterministic", "solver"})

    @property
    def input_specs(self) -> dict[str, TensorSpec]:
        return {
            "A": TensorSpec(
                name="A",
                kind=TensorKind.MATRIX,
                shape=(-1, -1),
                description="Coefficient matrix (m x n)",
            ),
            "b": TensorSpec(
                name="b",
                kind=TensorKind.VECTOR,
                description="Right-hand side vector (m,)",
            ),
        }

    @property
    def output_specs(self) -> dict[str, TensorSpec]:
        return {
            "x": TensorSpec(
                name="x",
                kind=TensorKind.VECTOR,
                description="Solution vector (n,)",
            ),
            "residual": TensorSpec(
                name="residual",
                kind=TensorKind.VECTOR,
                description="Residual vector b - Ax",
            ),
            "residual_norm": TensorSpec(
                name="residual_norm",
                description="L2 norm of residual",
            ),
        }

    def forward(
        self, inputs: dict[str, TrackedTensor]
    ) -> dict[str, TrackedTensor]:
        A = inputs["A"]
        b = inputs["b"]

        # Use lstsq for robust least squares solution
        x, residuals, rank, s = np.linalg.lstsq(A.data, b.data, rcond=None)

        # Compute residual vector
        residual = b.data - A.data @ x
        residual_norm = np.linalg.norm(residual)

        # Tags for solution
        x_tags: set[str] = {"least_squares_solution"}
        if A.data.shape[0] == A.data.shape[1]:
            x_tags.add("exact_solution")

        return {
            "x": TrackedTensor(
                data=x,
                name=f"x*({A.name},{b.name})",
                kind=TensorKind.VECTOR,
                tags=frozenset(x_tags),
            ),
            "residual": TrackedTensor(
                data=residual,
                name=f"r({A.name},{b.name})",
                kind=TensorKind.VECTOR,
                tags=frozenset({"residual"}),
            ),
            "residual_norm": TrackedTensor(
                data=np.array(residual_norm),
                name=f"||r||",
                kind=TensorKind.VECTOR,
                tags=frozenset({"scalar", "non_negative"}),
            ),
        }


class LinearSolve(Operator):
    """Direct linear system solver for square systems.

    Solves Ax = b where A is a square invertible matrix.
    Uses LU decomposition internally.
    """

    @property
    def name(self) -> str:
        return "LinearSolve"

    @property
    def description(self) -> str:
        return "Linear system solver: x = A^{-1}b (for square A)"

    @property
    def tags(self) -> frozenset[str]:
        return frozenset({"deterministic", "solver"})

    @property
    def input_specs(self) -> dict[str, TensorSpec]:
        return {
            "A": TensorSpec(
                name="A",
                kind=TensorKind.MATRIX,
                shape=(-1, -1),
                description="Square coefficient matrix (n x n)",
            ),
            "b": TensorSpec(
                name="b",
                kind=TensorKind.VECTOR,
                description="Right-hand side vector (n,)",
            ),
        }

    @property
    def output_specs(self) -> dict[str, TensorSpec]:
        return {
            "x": TensorSpec(
                name="x",
                kind=TensorKind.VECTOR,
                description="Solution vector (n,)",
            ),
        }

    def forward(
        self, inputs: dict[str, TrackedTensor]
    ) -> dict[str, TrackedTensor]:
        A = inputs["A"]
        b = inputs["b"]

        # Validate square matrix
        if A.data.shape[0] != A.data.shape[1]:
            raise ValueError(
                f"LinearSolve requires square matrix, got shape {A.data.shape}"
            )

        # Use solve for direct solution
        x = np.linalg.solve(A.data, b.data)

        return {
            "x": TrackedTensor(
                data=x,
                name=f"solve({A.name},{b.name})",
                kind=TensorKind.VECTOR,
                tags=frozenset({"exact_solution"}),
            ),
        }


class NormalEquations(Operator):
    """Solve least squares via normal equations.

    Solves (A^T A) x = A^T b, which is equivalent to the least squares
    problem min_x ||Ax - b||_2.

    This is useful for visualization as it exposes the intermediate
    A^T A and A^T b quantities.
    """

    @property
    def name(self) -> str:
        return "NormalEquations"

    @property
    def description(self) -> str:
        return "Normal equations solver: (A^T A)x = A^T b"

    @property
    def tags(self) -> frozenset[str]:
        return frozenset({"deterministic", "solver"})

    @property
    def input_specs(self) -> dict[str, TensorSpec]:
        return {
            "A": TensorSpec(
                name="A",
                kind=TensorKind.MATRIX,
                shape=(-1, -1),
                description="Design matrix (m x n)",
            ),
            "b": TensorSpec(
                name="b",
                kind=TensorKind.VECTOR,
                description="Observation vector (m,)",
            ),
        }

    @property
    def output_specs(self) -> dict[str, TensorSpec]:
        return {
            "x": TensorSpec(
                name="x",
                kind=TensorKind.VECTOR,
                description="Solution vector (n,)",
            ),
            "AtA": TensorSpec(
                name="AtA",
                kind=TensorKind.MATRIX,
                description="Normal matrix A^T A",
            ),
            "Atb": TensorSpec(
                name="Atb",
                kind=TensorKind.VECTOR,
                description="A^T b vector",
            ),
        }

    def forward(
        self, inputs: dict[str, TrackedTensor]
    ) -> dict[str, TrackedTensor]:
        A = inputs["A"]
        b = inputs["b"]

        At = A.data.T
        AtA = At @ A.data
        Atb = At @ b.data

        # Solve the normal equations
        x = np.linalg.solve(AtA, Atb)

        # AtA is always symmetric and PSD
        ata_tags: set[str] = {"symmetric", "psd", "normal_matrix"}

        # Check positive definiteness
        try:
            eigvals = np.linalg.eigvalsh(AtA)
            if np.all(eigvals > 1e-10):
                ata_tags.add("positive_definite")
        except np.linalg.LinAlgError:
            pass

        return {
            "x": TrackedTensor(
                data=x,
                name=f"x*({A.name})",
                kind=TensorKind.VECTOR,
                tags=frozenset({"least_squares_solution"}),
            ),
            "AtA": TrackedTensor(
                data=AtA,
                name=f"{A.name}^T{A.name}",
                kind=TensorKind.MATRIX,
                tags=frozenset(ata_tags),
            ),
            "Atb": TrackedTensor(
                data=Atb,
                name=f"{A.name}^T{b.name}",
                kind=TensorKind.VECTOR,
                tags=frozenset(),
            ),
        }


class Inverse(Operator):
    """Matrix inverse operator.

    Computes A^{-1} for a square invertible matrix.
    """

    @property
    def name(self) -> str:
        return "Inverse"

    @property
    def description(self) -> str:
        return "Matrix inverse: B = A^{-1}"

    @property
    def tags(self) -> frozenset[str]:
        return frozenset({"deterministic"})

    @property
    def input_specs(self) -> dict[str, TensorSpec]:
        return {
            "A": TensorSpec(
                name="A",
                kind=TensorKind.MATRIX,
                shape=(-1, -1),
                description="Square invertible matrix",
            ),
        }

    @property
    def output_specs(self) -> dict[str, TensorSpec]:
        return {
            "A_inv": TensorSpec(
                name="A_inv",
                kind=TensorKind.MATRIX,
                description="Inverse matrix",
            ),
        }

    def forward(
        self, inputs: dict[str, TrackedTensor]
    ) -> dict[str, TrackedTensor]:
        A = inputs["A"]

        if A.data.shape[0] != A.data.shape[1]:
            raise ValueError(
                f"Inverse requires square matrix, got shape {A.data.shape}"
            )

        A_inv = np.linalg.inv(A.data)

        # Preserve symmetry if present
        tags: set[str] = {"inverse"}
        if "symmetric" in A.tags:
            tags.add("symmetric")
        if "positive_definite" in A.tags:
            tags.add("positive_definite")

        return {
            "A_inv": TrackedTensor(
                data=A_inv,
                name=f"{A.name}^(-1)",
                kind=TensorKind.MATRIX,
                tags=frozenset(tags),
            ),
        }


class PseudoInverse(Operator):
    """Moore-Penrose pseudoinverse operator.

    Computes A^+ (the pseudoinverse) using SVD.
    Works for any matrix, including non-square and rank-deficient.
    """

    def __init__(self, rcond: float | None = None):
        """Initialize pseudoinverse operator.

        Args:
            rcond: Cutoff for small singular values. Singular values
                   less than rcond * largest_singular_value are set to zero.
                   If None, uses machine precision * max(m, n).
        """
        self._rcond = rcond

    @property
    def name(self) -> str:
        return "PseudoInverse"

    @property
    def description(self) -> str:
        return "Moore-Penrose pseudoinverse: A^+"

    @property
    def tags(self) -> frozenset[str]:
        return frozenset({"deterministic"})

    @property
    def input_specs(self) -> dict[str, TensorSpec]:
        return {
            "A": TensorSpec(
                name="A",
                kind=TensorKind.MATRIX,
                shape=(-1, -1),
                description="Input matrix (any shape)",
            ),
        }

    @property
    def output_specs(self) -> dict[str, TensorSpec]:
        return {
            "A_pinv": TensorSpec(
                name="A_pinv",
                kind=TensorKind.MATRIX,
                description="Pseudoinverse matrix",
            ),
        }

    def forward(
        self, inputs: dict[str, TrackedTensor]
    ) -> dict[str, TrackedTensor]:
        A = inputs["A"]

        if self._rcond is None:
            A_pinv = np.linalg.pinv(A.data)
        else:
            A_pinv = np.linalg.pinv(A.data, rcond=self._rcond)

        return {
            "A_pinv": TrackedTensor(
                data=A_pinv,
                name=f"{A.name}^+",
                kind=TensorKind.MATRIX,
                tags=frozenset({"pseudoinverse"}),
            ),
        }
