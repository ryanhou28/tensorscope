"""Scenario definition and execution for Tensorscope."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
from enum import Enum
import numpy as np

from .tensor import TrackedTensor, TensorKind, TensorSummary, compute_summary
from .graph import OperatorGraph


class ParameterType(Enum):
    """Types of parameters that can be adjusted in scenarios."""

    CONTINUOUS = "continuous"  # Slider with min/max
    DISCRETE = "discrete"  # Dropdown with options


@dataclass
class Parameter:
    """A tunable parameter in a scenario.

    Parameters allow users to interactively adjust scenario inputs
    and see how they affect the computation.

    Attributes:
        name: Unique identifier for this parameter.
        display_name: Human-readable name for UI display.
        param_type: Whether continuous (slider) or discrete (dropdown).
        default: Default value.
        min_val: Minimum value (for continuous parameters).
        max_val: Maximum value (for continuous parameters).
        step: Step size for slider (for continuous parameters).
        options: List of valid values (for discrete parameters).
        description: Help text for the parameter.
    """

    name: str
    display_name: str = ""
    param_type: ParameterType = ParameterType.CONTINUOUS
    default: float | int | str = 0.0
    min_val: float | None = None
    max_val: float | None = None
    step: float | None = None
    options: list[Any] | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not self.display_name:
            self.display_name = self.name.replace("_", " ").title()

    def validate(self, value: Any) -> tuple[bool, str]:
        """Validate a parameter value.

        Args:
            value: The value to validate.

        Returns:
            A tuple of (is_valid, error_message).
        """
        if self.param_type == ParameterType.CONTINUOUS:
            if not isinstance(value, (int, float)):
                return False, f"Expected numeric value, got {type(value).__name__}"
            if self.min_val is not None and value < self.min_val:
                return False, f"Value {value} is below minimum {self.min_val}"
            if self.max_val is not None and value > self.max_val:
                return False, f"Value {value} is above maximum {self.max_val}"
        elif self.param_type == ParameterType.DISCRETE:
            if self.options is not None and value not in self.options:
                return False, f"Value {value} not in options {self.options}"
        return True, ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize parameter for API responses."""
        result = {
            "name": self.name,
            "display_name": self.display_name,
            "type": self.param_type.value,
            "default": self.default,
            "description": self.description,
        }
        if self.param_type == ParameterType.CONTINUOUS:
            result["min"] = self.min_val
            result["max"] = self.max_val
            result["step"] = self.step
        elif self.param_type == ParameterType.DISCRETE:
            result["options"] = self.options
        return result


@dataclass
class ProbePoint:
    """A tensor marked for inspection in the UI.

    Attributes:
        tensor_key: The key identifying this tensor (e.g., "node.output").
        display_name: Human-readable name for display.
        description: Help text explaining what this tensor represents.
    """

    tensor_key: str
    display_name: str = ""
    description: str = ""


class Scenario:
    """A self-contained linear algebra demonstration.

    Scenarios define:
    - Input generation based on parameters
    - An operator graph that processes the inputs
    - Probe points marking tensors of interest for visualization

    Example:
        >>> scenario = Scenario(
        ...     name="least_squares_2d",
        ...     description="2D least squares projection demo"
        ... )
        >>> scenario.param("noise_level", min_val=0.0, max_val=1.0, default=0.1)
        >>> scenario.param("condition_number", min_val=1.0, max_val=100.0, default=10.0)
        >>>
        >>> # Set up input generator
        >>> def generate_inputs(params):
        ...     # Return dict of TrackedTensors based on params
        ...     pass
        >>> scenario.set_input_generator(generate_inputs)
        >>>
        >>> # Set up operator graph
        >>> graph = OperatorGraph()
        >>> # ... add nodes and connections ...
        >>> scenario.set_graph(graph)
        >>>
        >>> # Mark tensors for inspection
        >>> scenario.probe("node.output", display_name="Result")
        >>>
        >>> # Run with specific parameters
        >>> results = scenario.run({"noise_level": 0.2})
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        id: str | None = None,
    ) -> None:
        """Initialize a scenario.

        Args:
            name: Human-readable name for this scenario.
            description: Longer description of what the scenario demonstrates.
            id: Unique identifier (defaults to name with underscores).
        """
        self.name = name
        self.description = description
        self.id = id or name.lower().replace(" ", "_")

        self._parameters: dict[str, Parameter] = {}
        self._probes: list[ProbePoint] = []
        self._graph: OperatorGraph | None = None
        self._input_generator: Callable[[dict[str, Any]], dict[str, TrackedTensor]] | None = None
        self._last_results: dict[str, TrackedTensor] = {}
        self._last_params: dict[str, Any] = {}

    def param(
        self,
        name: str,
        *,
        display_name: str = "",
        min_val: float | None = None,
        max_val: float | None = None,
        default: float | int | str = 0.0,
        step: float | None = None,
        options: list[Any] | None = None,
        description: str = "",
    ) -> Parameter:
        """Define a tunable parameter for this scenario.

        Args:
            name: Unique identifier for the parameter.
            display_name: Human-readable name (defaults to formatted name).
            min_val: Minimum value (for sliders).
            max_val: Maximum value (for sliders).
            default: Default value.
            step: Step size for slider.
            options: List of valid values (for dropdowns).
            description: Help text.

        Returns:
            The created Parameter object.
        """
        if options is not None:
            param_type = ParameterType.DISCRETE
        else:
            param_type = ParameterType.CONTINUOUS

        param = Parameter(
            name=name,
            display_name=display_name,
            param_type=param_type,
            default=default,
            min_val=min_val,
            max_val=max_val,
            step=step,
            options=options,
            description=description,
        )
        self._parameters[name] = param
        return param

    def probe(
        self,
        tensor_key: str,
        *,
        display_name: str = "",
        description: str = "",
    ) -> ProbePoint:
        """Mark a tensor for inspection in the UI.

        Args:
            tensor_key: The key identifying the tensor (e.g., "matmul.C").
            display_name: Human-readable name for display.
            description: Help text.

        Returns:
            The created ProbePoint object.
        """
        probe = ProbePoint(
            tensor_key=tensor_key,
            display_name=display_name or tensor_key,
            description=description,
        )
        self._probes.append(probe)
        return probe

    def set_graph(self, graph: OperatorGraph) -> None:
        """Set the operator graph for this scenario.

        Args:
            graph: The configured OperatorGraph.
        """
        self._graph = graph

    def set_input_generator(
        self,
        generator: Callable[[dict[str, Any]], dict[str, TrackedTensor]],
    ) -> None:
        """Set the function that generates input tensors from parameters.

        Args:
            generator: A function that takes parameter values and returns
                       a dict mapping input names to TrackedTensors.
        """
        self._input_generator = generator

    @property
    def parameters(self) -> dict[str, Parameter]:
        """Get all defined parameters."""
        return self._parameters.copy()

    @property
    def probes(self) -> list[ProbePoint]:
        """Get all probe points."""
        return self._probes.copy()

    @property
    def graph(self) -> OperatorGraph | None:
        """Get the operator graph."""
        return self._graph

    def get_default_params(self) -> dict[str, Any]:
        """Get default values for all parameters.

        Returns:
            A dict mapping parameter names to their default values.
        """
        return {name: param.default for name, param in self._parameters.items()}

    def validate_params(self, params: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate parameter values.

        Args:
            params: The parameter values to validate.

        Returns:
            A tuple of (all_valid, list_of_errors).
        """
        errors: list[str] = []
        for name, value in params.items():
            if name not in self._parameters:
                errors.append(f"Unknown parameter: {name}")
                continue
            is_valid, error = self._parameters[name].validate(value)
            if not is_valid:
                errors.append(f"{name}: {error}")
        return len(errors) == 0, errors

    def run(
        self,
        params: dict[str, Any] | None = None,
    ) -> dict[str, TrackedTensor]:
        """Execute the scenario with given parameters.

        Args:
            params: Parameter values. Missing parameters use defaults.

        Returns:
            A dict mapping tensor keys to TrackedTensors for all outputs.

        Raises:
            ValueError: If the scenario is not properly configured.
        """
        if self._graph is None:
            raise ValueError("Scenario graph not set. Call set_graph() first.")
        if self._input_generator is None:
            raise ValueError("Input generator not set. Call set_input_generator() first.")

        # Merge with defaults
        full_params = self.get_default_params()
        if params:
            full_params.update(params)

        # Validate
        is_valid, errors = self.validate_params(full_params)
        if not is_valid:
            raise ValueError(f"Invalid parameters: {errors}")

        # Generate inputs
        inputs = self._input_generator(full_params)

        # Execute graph
        results = self._graph.execute(inputs)

        # Cache results
        self._last_results = results
        self._last_params = full_params

        return results

    def get_probed_tensors(self) -> dict[str, TrackedTensor]:
        """Get only the probed tensors from the last execution.

        Returns:
            A dict mapping probe display names to TrackedTensors.
        """
        probed: dict[str, TrackedTensor] = {}
        for probe in self._probes:
            if probe.tensor_key in self._last_results:
                probed[probe.display_name] = self._last_results[probe.tensor_key]
        return probed

    def get_probed_summaries(self) -> dict[str, TensorSummary]:
        """Get summaries of probed tensors from the last execution.

        Returns:
            A dict mapping probe display names to TensorSummaries.
        """
        summaries: dict[str, TensorSummary] = {}
        for probe in self._probes:
            if probe.tensor_key in self._last_results:
                tensor = self._last_results[probe.tensor_key]
                summaries[probe.display_name] = compute_summary(tensor)
        return summaries

    def to_dict(self) -> dict[str, Any]:
        """Serialize scenario metadata for API responses.

        Returns:
            A dict with scenario info suitable for JSON serialization.
        """
        result: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "parameters": [p.to_dict() for p in self._parameters.values()],
            "probes": [
                {
                    "key": p.tensor_key,
                    "display_name": p.display_name,
                    "description": p.description,
                }
                for p in self._probes
            ],
        }
        if self._graph:
            result["graph"] = self._graph.to_dict()
        return result

    def __repr__(self) -> str:
        return (
            f"Scenario(id={self.id!r}, name={self.name!r}, "
            f"params={len(self._parameters)}, probes={len(self._probes)})"
        )
