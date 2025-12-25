"""Basic linear algebra operators for Tensorscope."""

from __future__ import annotations

import numpy as np

from ..core.operator import Operator, TensorSpec
from ..core.tensor import TrackedTensor, TensorKind


class MatMul(Operator):
    """Matrix multiplication operator.

    Computes A @ B where A and B are matrices (or matrix-vector products).
    """

    @property
    def name(self) -> str:
        return "MatMul"

    @property
    def description(self) -> str:
        return "Matrix multiplication: C = A @ B"

    @property
    def tags(self) -> frozenset[str]:
        return frozenset({"linear", "deterministic", "differentiable"})

    @property
    def input_specs(self) -> dict[str, TensorSpec]:
        return {
            "A": TensorSpec(
                name="A",
                kind=TensorKind.MATRIX,
                shape=(-1, -1),
                description="Left matrix",
            ),
            "B": TensorSpec(
                name="B",
                description="Right matrix or vector",
            ),
        }

    @property
    def output_specs(self) -> dict[str, TensorSpec]:
        return {
            "C": TensorSpec(
                name="C",
                description="Result of A @ B",
            ),
        }

    def forward(
        self, inputs: dict[str, TrackedTensor]
    ) -> dict[str, TrackedTensor]:
        A = inputs["A"]
        B = inputs["B"]

        result = A.data @ B.data

        # Determine output kind
        if result.ndim == 1:
            kind = TensorKind.VECTOR
        else:
            kind = TensorKind.MATRIX

        # Determine output tags
        tags: set[str] = set()

        # Check if result is symmetric (only for square matrices)
        if result.ndim == 2 and result.shape[0] == result.shape[1]:
            if np.allclose(result, result.T):
                tags.add("symmetric")
                # Check positive definiteness
                try:
                    eigvals = np.linalg.eigvalsh(result)
                    if np.all(eigvals > 0):
                        tags.add("psd")
                        tags.add("positive_definite")
                    elif np.all(eigvals >= -1e-10):
                        tags.add("psd")
                except np.linalg.LinAlgError:
                    pass

        return {
            "C": TrackedTensor(
                data=result,
                name=f"{A.name}@{B.name}",
                kind=kind,
                tags=frozenset(tags),
            )
        }


class Transpose(Operator):
    """Matrix transpose operator.

    Computes A^T for a matrix A.
    """

    @property
    def name(self) -> str:
        return "Transpose"

    @property
    def description(self) -> str:
        return "Matrix transpose: B = A^T"

    @property
    def tags(self) -> frozenset[str]:
        return frozenset({"linear", "deterministic", "differentiable"})

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
            "At": TensorSpec(
                name="At",
                kind=TensorKind.MATRIX,
                description="Transposed matrix",
            ),
        }

    def forward(
        self, inputs: dict[str, TrackedTensor]
    ) -> dict[str, TrackedTensor]:
        A = inputs["A"]
        result = A.data.T

        # Preserve symmetry tag if present
        tags = set(A.tags)

        return {
            "At": TrackedTensor(
                data=result,
                name=f"{A.name}^T",
                kind=TensorKind.MATRIX,
                tags=frozenset(tags),
            )
        }


class Norm(Operator):
    """Norm computation operator.

    Computes various norms: Frobenius (default), L1, L2, Linf.
    """

    def __init__(self, ord: str = "fro"):
        """Initialize with norm type.

        Args:
            ord: Norm type. One of 'fro', 'l1', 'l2', 'linf', 'nuc'.
        """
        self._ord = ord

    @property
    def name(self) -> str:
        return f"Norm({self._ord})"

    @property
    def description(self) -> str:
        norm_names = {
            "fro": "Frobenius",
            "l1": "L1",
            "l2": "L2 (spectral)",
            "linf": "L-infinity",
            "nuc": "Nuclear",
        }
        return f"{norm_names.get(self._ord, self._ord)} norm"

    @property
    def tags(self) -> frozenset[str]:
        return frozenset({"deterministic"})

    @property
    def input_specs(self) -> dict[str, TensorSpec]:
        return {
            "A": TensorSpec(
                name="A",
                description="Input tensor",
            ),
        }

    @property
    def output_specs(self) -> dict[str, TensorSpec]:
        return {
            "norm": TensorSpec(
                name="norm",
                description="Computed norm (scalar)",
            ),
        }

    def forward(
        self, inputs: dict[str, TrackedTensor]
    ) -> dict[str, TrackedTensor]:
        A = inputs["A"]

        ord_map = {
            "fro": "fro",
            "l1": 1,
            "l2": 2,
            "linf": np.inf,
            "nuc": "nuc",
        }
        numpy_ord = ord_map.get(self._ord, "fro")

        norm_value = np.linalg.norm(A.data, ord=numpy_ord)

        return {
            "norm": TrackedTensor(
                data=np.array(norm_value),
                name=f"||{A.name}||_{self._ord}",
                kind=TensorKind.VECTOR,  # Scalar treated as 0-d vector
                tags=frozenset({"scalar", "non_negative"}),
            )
        }


class Add(Operator):
    """Element-wise addition operator.

    Computes A + B with broadcasting support.
    """

    @property
    def name(self) -> str:
        return "Add"

    @property
    def description(self) -> str:
        return "Element-wise addition: C = A + B"

    @property
    def tags(self) -> frozenset[str]:
        return frozenset({"linear", "deterministic", "differentiable"})

    @property
    def input_specs(self) -> dict[str, TensorSpec]:
        return {
            "A": TensorSpec(name="A", description="First operand"),
            "B": TensorSpec(name="B", description="Second operand"),
        }

    @property
    def output_specs(self) -> dict[str, TensorSpec]:
        return {
            "C": TensorSpec(name="C", description="Sum A + B"),
        }

    def forward(
        self, inputs: dict[str, TrackedTensor]
    ) -> dict[str, TrackedTensor]:
        A = inputs["A"]
        B = inputs["B"]

        result = A.data + B.data

        # Determine output kind
        if result.ndim == 1:
            kind = TensorKind.VECTOR
        elif result.ndim == 2:
            kind = TensorKind.MATRIX
        else:
            kind = A.kind  # Preserve input kind for higher dimensions

        # Determine tags
        tags: set[str] = set()
        if result.ndim == 2 and result.shape[0] == result.shape[1]:
            if np.allclose(result, result.T):
                tags.add("symmetric")

        return {
            "C": TrackedTensor(
                data=result,
                name=f"({A.name}+{B.name})",
                kind=kind,
                tags=frozenset(tags),
            )
        }


class Subtract(Operator):
    """Element-wise subtraction operator.

    Computes A - B with broadcasting support.
    """

    @property
    def name(self) -> str:
        return "Subtract"

    @property
    def description(self) -> str:
        return "Element-wise subtraction: C = A - B"

    @property
    def tags(self) -> frozenset[str]:
        return frozenset({"linear", "deterministic", "differentiable"})

    @property
    def input_specs(self) -> dict[str, TensorSpec]:
        return {
            "A": TensorSpec(name="A", description="First operand (minuend)"),
            "B": TensorSpec(name="B", description="Second operand (subtrahend)"),
        }

    @property
    def output_specs(self) -> dict[str, TensorSpec]:
        return {
            "C": TensorSpec(name="C", description="Difference A - B"),
        }

    def forward(
        self, inputs: dict[str, TrackedTensor]
    ) -> dict[str, TrackedTensor]:
        A = inputs["A"]
        B = inputs["B"]

        result = A.data - B.data

        # Determine output kind
        if result.ndim == 1:
            kind = TensorKind.VECTOR
        elif result.ndim == 2:
            kind = TensorKind.MATRIX
        else:
            kind = A.kind

        # Determine tags
        tags: set[str] = set()
        if result.ndim == 2 and result.shape[0] == result.shape[1]:
            if np.allclose(result, result.T):
                tags.add("symmetric")

        return {
            "C": TrackedTensor(
                data=result,
                name=f"({A.name}-{B.name})",
                kind=kind,
                tags=frozenset(tags),
            )
        }


class Scale(Operator):
    """Scalar multiplication operator.

    Computes alpha * A where alpha is a scalar.
    """

    def __init__(self, alpha: float = 1.0):
        """Initialize with scale factor.

        Args:
            alpha: The scalar multiplier.
        """
        self._alpha = alpha

    @property
    def name(self) -> str:
        return f"Scale({self._alpha})"

    @property
    def description(self) -> str:
        return f"Scalar multiplication: B = {self._alpha} * A"

    @property
    def tags(self) -> frozenset[str]:
        return frozenset({"linear", "deterministic", "differentiable"})

    @property
    def input_specs(self) -> dict[str, TensorSpec]:
        return {
            "A": TensorSpec(name="A", description="Input tensor"),
        }

    @property
    def output_specs(self) -> dict[str, TensorSpec]:
        return {
            "B": TensorSpec(name="B", description="Scaled tensor"),
        }

    def forward(
        self, inputs: dict[str, TrackedTensor]
    ) -> dict[str, TrackedTensor]:
        A = inputs["A"]
        result = self._alpha * A.data

        # Preserve tags (scaling preserves symmetry, etc.)
        tags = set(A.tags)

        # Scaling by negative flips positive definiteness
        if self._alpha < 0:
            if "positive_definite" in tags:
                tags.discard("positive_definite")
                tags.add("negative_definite")
            elif "negative_definite" in tags:
                tags.discard("negative_definite")
                tags.add("positive_definite")

        return {
            "B": TrackedTensor(
                data=result,
                name=f"{self._alpha}*{A.name}",
                kind=A.kind,
                tags=frozenset(tags),
            )
        }
