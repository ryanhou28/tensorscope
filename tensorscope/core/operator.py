"""Base operator interface for Tensorscope."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .tensor import TrackedTensor, TensorKind, TensorSummary, compute_summary


@dataclass
class TensorSpec:
    """Specification for an operator input or output tensor.

    Used to define the expected shape and kind of tensors that an operator
    accepts or produces.

    Attributes:
        name: The name of this input/output slot.
        kind: Expected tensor kind (or None for any).
        shape: Expected shape constraints. Use -1 for variable dimensions.
               e.g., (-1, -1) means any 2D matrix, (3, -1) means 3 rows, any columns.
        optional: Whether this input is optional.
        description: Human-readable description of this tensor.
    """

    name: str
    kind: TensorKind | None = None
    shape: tuple[int, ...] | None = None
    optional: bool = False
    description: str = ""

    def validate(self, tensor: TrackedTensor | None) -> tuple[bool, str]:
        """Validate a tensor against this spec.

        Args:
            tensor: The tensor to validate, or None for missing optional inputs.

        Returns:
            A tuple of (is_valid, error_message).
        """
        if tensor is None:
            if self.optional:
                return True, ""
            return False, f"Required input '{self.name}' is missing"

        if self.kind is not None and tensor.kind != self.kind:
            return False, (
                f"Input '{self.name}' expected kind {self.kind.value}, "
                f"got {tensor.kind.value}"
            )

        if self.shape is not None:
            if len(tensor.shape) != len(self.shape):
                return False, (
                    f"Input '{self.name}' expected {len(self.shape)}D tensor, "
                    f"got {len(tensor.shape)}D"
                )
            for i, (expected, actual) in enumerate(zip(self.shape, tensor.shape)):
                if expected != -1 and expected != actual:
                    return False, (
                        f"Input '{self.name}' expected dimension {i} to be {expected}, "
                        f"got {actual}"
                    )

        return True, ""


class Operator(ABC):
    """Abstract base class for all operators in Tensorscope.

    Operators are the building blocks of operator graphs. Each operator
    takes named input tensors and produces named output tensors.

    Subclasses must implement:
        - forward(): Execute the operation
        - input_specs: Define expected inputs
        - output_specs: Define expected outputs
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the operator name for display."""
        pass

    @property
    @abstractmethod
    def input_specs(self) -> dict[str, TensorSpec]:
        """Return specifications for input tensors.

        Returns:
            A dict mapping input names to their specifications.
        """
        pass

    @property
    @abstractmethod
    def output_specs(self) -> dict[str, TensorSpec]:
        """Return specifications for output tensors.

        Returns:
            A dict mapping output names to their specifications.
        """
        pass

    @property
    def tags(self) -> frozenset[str]:
        """Return semantic tags for this operator.

        Common tags include:
            - 'linear': The operator is linear
            - 'deterministic': Same inputs always produce same outputs
            - 'differentiable': Jacobian can be computed

        Returns:
            A frozenset of tag strings.
        """
        return frozenset()

    @property
    def description(self) -> str:
        """Return a human-readable description of the operator."""
        return ""

    @abstractmethod
    def forward(
        self, inputs: dict[str, TrackedTensor]
    ) -> dict[str, TrackedTensor]:
        """Execute the operator on input tensors.

        Args:
            inputs: A dict mapping input names to TrackedTensors.

        Returns:
            A dict mapping output names to TrackedTensors.

        Raises:
            ValueError: If inputs don't match input_specs.
        """
        pass

    def summarize(self, tensor: TrackedTensor) -> TensorSummary:
        """Generate summary statistics for a tensor.

        The default implementation uses compute_summary(). Subclasses can
        override to add operator-specific statistics.

        Args:
            tensor: The tensor to summarize.

        Returns:
            A TensorSummary with statistics.
        """
        return compute_summary(tensor)

    def jacobian(
        self,
        inputs: dict[str, TrackedTensor],
        output_name: str,
        input_name: str,
    ) -> TrackedTensor | None:
        """Compute the Jacobian of an output with respect to an input.

        The default implementation returns None (Jacobian not available).
        Subclasses can override to provide analytical or auto-diff Jacobians.

        Args:
            inputs: The input tensors.
            output_name: The name of the output tensor.
            input_name: The name of the input tensor to differentiate with respect to.

        Returns:
            A TrackedTensor containing the Jacobian, or None if not available.
        """
        return None

    def validate_inputs(self, inputs: dict[str, TrackedTensor]) -> None:
        """Validate input tensors against input_specs.

        Args:
            inputs: The input tensors to validate.

        Raises:
            ValueError: If any input doesn't match its spec.
        """
        for name, spec in self.input_specs.items():
            tensor = inputs.get(name)
            is_valid, error = spec.validate(tensor)
            if not is_valid:
                raise ValueError(f"Operator '{self.name}': {error}")

        # Check for unexpected inputs
        expected_names = set(self.input_specs.keys())
        actual_names = set(inputs.keys())
        unexpected = actual_names - expected_names
        if unexpected:
            raise ValueError(
                f"Operator '{self.name}': Unexpected inputs: {unexpected}"
            )

    def __call__(
        self, inputs: dict[str, TrackedTensor]
    ) -> dict[str, TrackedTensor]:
        """Execute the operator (validates inputs first).

        This is a convenience wrapper around forward() that validates inputs.

        Args:
            inputs: A dict mapping input names to TrackedTensors.

        Returns:
            A dict mapping output names to TrackedTensors.
        """
        self.validate_inputs(inputs)
        return self.forward(inputs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


@dataclass
class OperatorInstance:
    """An operator instance in a graph with a unique node name.

    This wraps an Operator with graph-specific metadata.

    Attributes:
        operator: The underlying operator.
        node_name: Unique name for this node in the graph.
        config: Optional configuration dict for parameterized operators.
    """

    operator: Operator
    node_name: str
    config: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"OperatorInstance(node={self.node_name!r}, op={self.operator.name!r})"
