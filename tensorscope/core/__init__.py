"""Core abstractions for Tensorscope."""

from .tensor import (
    TensorKind,
    TrackedTensor,
    TensorSummary,
    compute_summary,
)

from .operator import (
    TensorSpec,
    Operator,
    OperatorInstance,
)

from .graph import (
    Edge,
    OperatorGraph,
)

from .registry import (
    OperatorRegistry,
    register_operator,
)

__all__ = [
    # tensor.py
    "TensorKind",
    "TrackedTensor",
    "TensorSummary",
    "compute_summary",
    # operator.py
    "TensorSpec",
    "Operator",
    "OperatorInstance",
    # graph.py
    "Edge",
    "OperatorGraph",
    # registry.py
    "OperatorRegistry",
    "register_operator",
]
