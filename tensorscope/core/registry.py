"""Global registries for operators and visualizers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Type

if TYPE_CHECKING:
    from .operator import Operator
    from .tensor import TrackedTensor, TensorKind


class OperatorRegistry:
    """Singleton registry for operator classes.

    Provides a central place to register and discover operators.
    Use the @register decorator to register operator classes.

    Example:
        >>> @OperatorRegistry.register
        ... class MyOperator(Operator):
        ...     name = "my_op"
        ...     ...
        >>>
        >>> # Later, get all registered operators
        >>> ops = OperatorRegistry.get_all()
    """

    _instance: "OperatorRegistry | None" = None
    _operators: dict[str, Type["Operator"]] = {}
    _visualizer_rules: list[
        tuple[Callable[["TrackedTensor"], bool], list[str]]
    ] = []

    def __new__(cls) -> "OperatorRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, operator_class: Type["Operator"]) -> Type["Operator"]:
        """Register an operator class.

        Can be used as a decorator:
            @OperatorRegistry.register
            class MyOperator(Operator):
                ...

        Args:
            operator_class: The operator class to register.

        Returns:
            The operator class (unchanged).

        Raises:
            ValueError: If an operator with this name is already registered.
        """
        # Create a temporary instance to get the name
        # This is a bit hacky but avoids requiring a class attribute
        try:
            instance = operator_class.__new__(operator_class)
            name = instance.name
        except Exception:
            # If we can't instantiate, use the class name
            name = operator_class.__name__

        if name in cls._operators:
            raise ValueError(f"Operator '{name}' is already registered")

        cls._operators[name] = operator_class
        return operator_class

    @classmethod
    def get(cls, name: str) -> Type["Operator"] | None:
        """Get an operator class by name.

        Args:
            name: The operator name.

        Returns:
            The operator class, or None if not found.
        """
        return cls._operators.get(name)

    @classmethod
    def get_all(cls) -> dict[str, Type["Operator"]]:
        """Get all registered operator classes.

        Returns:
            A dict mapping operator names to classes.
        """
        return cls._operators.copy()

    @classmethod
    def clear(cls) -> None:
        """Clear all registered operators. Useful for testing."""
        cls._operators.clear()

    @classmethod
    def add_visualizer_rule(
        cls,
        predicate: Callable[["TrackedTensor"], bool],
        visualizers: list[str],
    ) -> None:
        """Add a rule for recommending visualizers based on tensor properties.

        Args:
            predicate: A function that takes a TrackedTensor and returns True
                      if the visualizers should be recommended.
            visualizers: A list of visualizer names to recommend.
        """
        cls._visualizer_rules.append((predicate, visualizers))

    @classmethod
    def get_recommended_visualizers(cls, tensor: "TrackedTensor") -> list[str]:
        """Get recommended visualizers for a tensor.

        Checks all registered rules and returns matching visualizers.
        Also includes default recommendations based on tensor kind.

        Args:
            tensor: The tensor to get recommendations for.

        Returns:
            A list of recommended visualizer names.
        """
        from .tensor import TensorKind

        visualizers: list[str] = []

        # Default recommendations based on kind
        kind_defaults = {
            TensorKind.VECTOR: ["vector_stem", "bar_chart"],
            TensorKind.MATRIX: ["heatmap"],
            TensorKind.IMAGE: ["image"],
            TensorKind.SPARSE_MATRIX: ["sparsity_pattern", "heatmap"],
            TensorKind.POINTCLOUD: ["scatter_2d", "scatter_3d"],
        }

        if tensor.kind in kind_defaults:
            visualizers.extend(kind_defaults[tensor.kind])

        # Apply custom rules
        for predicate, viz_list in cls._visualizer_rules:
            try:
                if predicate(tensor):
                    for v in viz_list:
                        if v not in visualizers:
                            visualizers.append(v)
            except Exception:
                # Ignore rule failures
                pass

        return visualizers


# Convenience function for decorator usage
def register_operator(operator_class: Type["Operator"]) -> Type["Operator"]:
    """Decorator to register an operator class.

    Example:
        >>> @register_operator
        ... class MyOperator(Operator):
        ...     ...
    """
    return OperatorRegistry.register(operator_class)
