"""Operator graph for building and executing computation DAGs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .tensor import TrackedTensor
from .operator import Operator, OperatorInstance


@dataclass
class Edge:
    """An edge in the operator graph, representing data flow.

    Attributes:
        from_node: The source node name.
        from_output: The output name on the source node.
        to_node: The destination node name.
        to_input: The input name on the destination node.
    """

    from_node: str
    from_output: str
    to_node: str
    to_input: str

    def __repr__(self) -> str:
        return f"{self.from_node}.{self.from_output} -> {self.to_node}.{self.to_input}"


class OperatorGraph:
    """A directed acyclic graph of operators.

    The graph manages operator nodes and their connections, and executes
    them in topological order.

    Example:
        >>> graph = OperatorGraph()
        >>> graph.add_node(MatMulOperator(), "matmul1")
        >>> graph.add_node(TransposeOperator(), "transpose1")
        >>> graph.connect("input", "A", "transpose1", "input")
        >>> graph.connect("transpose1", "output", "matmul1", "A")
        >>> graph.connect("input", "A", "matmul1", "B")
        >>> results = graph.execute({"A": tensor_a})
    """

    def __init__(self) -> None:
        """Initialize an empty operator graph."""
        self._nodes: dict[str, OperatorInstance] = {}
        self._edges: list[Edge] = []
        self._input_node = "_input"
        self._cached_order: list[str] | None = None
        self._last_execution_tensors: dict[str, TrackedTensor] = {}

    def add_node(
        self,
        operator: Operator,
        name: str,
        config: dict[str, Any] | None = None,
    ) -> str:
        """Add an operator node to the graph.

        Args:
            operator: The operator to add.
            name: A unique name for this node.
            config: Optional configuration for parameterized operators.

        Returns:
            The node name.

        Raises:
            ValueError: If a node with this name already exists.
        """
        if name in self._nodes or name == self._input_node:
            raise ValueError(f"Node '{name}' already exists in graph")

        self._nodes[name] = OperatorInstance(
            operator=operator,
            node_name=name,
            config=config or {},
        )
        self._cached_order = None
        return name

    def connect(
        self,
        from_node: str,
        from_output: str,
        to_node: str,
        to_input: str,
    ) -> None:
        """Connect an output of one node to an input of another.

        Use "_input" as from_node to connect graph inputs.

        Args:
            from_node: Source node name (or "_input" for graph inputs).
            from_output: Output name on the source node.
            to_node: Destination node name.
            to_input: Input name on the destination node.

        Raises:
            ValueError: If nodes don't exist or connection is invalid.
        """
        if from_node != self._input_node and from_node not in self._nodes:
            raise ValueError(f"Source node '{from_node}' not found")
        if to_node not in self._nodes:
            raise ValueError(f"Destination node '{to_node}' not found")

        # Validate output exists on source (skip for graph inputs)
        if from_node != self._input_node:
            source_op = self._nodes[from_node].operator
            if from_output not in source_op.output_specs:
                raise ValueError(
                    f"Node '{from_node}' has no output '{from_output}'. "
                    f"Available: {list(source_op.output_specs.keys())}"
                )

        # Validate input exists on destination
        dest_op = self._nodes[to_node].operator
        if to_input not in dest_op.input_specs:
            raise ValueError(
                f"Node '{to_node}' has no input '{to_input}'. "
                f"Available: {list(dest_op.input_specs.keys())}"
            )

        self._edges.append(Edge(from_node, from_output, to_node, to_input))
        self._cached_order = None

    def _topological_sort(self) -> list[str]:
        """Return nodes in topological order (dependencies first).

        Returns:
            A list of node names in execution order.

        Raises:
            ValueError: If the graph contains cycles.
        """
        if self._cached_order is not None:
            return self._cached_order

        # Build adjacency and in-degree
        in_degree: dict[str, int] = {name: 0 for name in self._nodes}
        adjacency: dict[str, list[str]] = defaultdict(list)

        for edge in self._edges:
            if edge.from_node != self._input_node:
                if edge.to_node not in adjacency[edge.from_node]:
                    adjacency[edge.from_node].append(edge.to_node)
                in_degree[edge.to_node] += 1

        # Adjust for duplicate edges (only count unique from_node -> to_node)
        in_degree = {name: 0 for name in self._nodes}
        unique_deps: dict[str, set[str]] = defaultdict(set)
        for edge in self._edges:
            if edge.from_node != self._input_node:
                unique_deps[edge.to_node].add(edge.from_node)
        for node, deps in unique_deps.items():
            in_degree[node] = len(deps)

        # Kahn's algorithm
        queue = [name for name, deg in in_degree.items() if deg == 0]
        result: list[str] = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for neighbor in adjacency[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(self._nodes):
            raise ValueError("Graph contains cycles")

        self._cached_order = result
        return result

    def execute(
        self, inputs: dict[str, TrackedTensor]
    ) -> dict[str, TrackedTensor]:
        """Execute the graph with given inputs.

        Args:
            inputs: A dict mapping input names to TrackedTensors.

        Returns:
            A dict mapping "<node>.<output>" to TrackedTensors for all
            outputs of all nodes.
        """
        execution_order = self._topological_sort()

        # Map of "node.output" -> tensor
        tensors: dict[str, TrackedTensor] = {}

        # Store graph inputs
        for name, tensor in inputs.items():
            tensors[f"{self._input_node}.{name}"] = tensor

        # Execute nodes in order
        for node_name in execution_order:
            instance = self._nodes[node_name]
            operator = instance.operator

            # Gather inputs for this node
            node_inputs: dict[str, TrackedTensor] = {}
            for edge in self._edges:
                if edge.to_node == node_name:
                    source_key = f"{edge.from_node}.{edge.from_output}"
                    if source_key not in tensors:
                        raise ValueError(
                            f"Missing tensor '{source_key}' for node '{node_name}'"
                        )
                    node_inputs[edge.to_input] = tensors[source_key]

            # Execute the operator
            outputs = operator(node_inputs)

            # Store outputs
            for output_name, tensor in outputs.items():
                tensors[f"{node_name}.{output_name}"] = tensor

        # Cache for get_all_tensors()
        self._last_execution_tensors = tensors
        return tensors

    def get_all_tensors(self) -> dict[str, TrackedTensor]:
        """Get all tensors from the last execution.

        Returns:
            A dict mapping "<node>.<output>" to TrackedTensors.
        """
        return self._last_execution_tensors.copy()

    def get_nodes(self) -> dict[str, OperatorInstance]:
        """Get all nodes in the graph.

        Returns:
            A dict mapping node names to OperatorInstances.
        """
        return self._nodes.copy()

    def get_edges(self) -> list[Edge]:
        """Get all edges in the graph.

        Returns:
            A list of Edge objects.
        """
        return self._edges.copy()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the graph structure for API responses.

        Returns:
            A dict with 'nodes' and 'edges' lists.
        """
        nodes = []
        for name, instance in self._nodes.items():
            nodes.append({
                "id": name,
                "name": instance.operator.name,
                "inputs": list(instance.operator.input_specs.keys()),
                "outputs": list(instance.operator.output_specs.keys()),
                "tags": sorted(instance.operator.tags),
            })

        edges = []
        for edge in self._edges:
            edges.append({
                "from_node": edge.from_node,
                "from_output": edge.from_output,
                "to_node": edge.to_node,
                "to_input": edge.to_input,
            })

        return {"nodes": nodes, "edges": edges}

    def __repr__(self) -> str:
        return f"OperatorGraph(nodes={len(self._nodes)}, edges={len(self._edges)})"
