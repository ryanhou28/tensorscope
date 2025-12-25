"""Linear algebra operators for Tensorscope."""

from .basic import (
    MatMul,
    Transpose,
    Norm,
    Add,
    Subtract,
    Scale,
)

from .decompositions import (
    SVD,
    Eigendecomposition,
    QR,
    Cholesky,
)

from .solvers import (
    LeastSquares,
    LinearSolve,
    NormalEquations,
    Inverse,
    PseudoInverse,
)

__all__ = [
    # Basic operators
    "MatMul",
    "Transpose",
    "Norm",
    "Add",
    "Subtract",
    "Scale",
    # Decompositions
    "SVD",
    "Eigendecomposition",
    "QR",
    "Cholesky",
    # Solvers
    "LeastSquares",
    "LinearSolve",
    "NormalEquations",
    "Inverse",
    "PseudoInverse",
]
