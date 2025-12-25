"""Tests for core types: TrackedTensor, Operator, OperatorGraph."""

import numpy as np
import pytest

from tensorscope.core import (
    TensorKind,
    TrackedTensor,
    TensorSummary,
    compute_summary,
    TensorSpec,
    Operator,
    OperatorGraph,
    OperatorRegistry,
    register_operator,
)


class TestTrackedTensor:
    """Tests for TrackedTensor."""

    def test_create_vector(self):
        """Can create a TrackedTensor with a vector."""
        data = np.array([1.0, 2.0, 3.0])
        tensor = TrackedTensor(
            data=data,
            name="my_vector",
            kind=TensorKind.VECTOR,
        )

        assert tensor.name == "my_vector"
        assert tensor.kind == TensorKind.VECTOR
        assert tensor.shape == (3,)
        assert tensor.dtype == "float64"
        assert np.array_equal(tensor.data, data)

    def test_create_matrix(self):
        """Can create a TrackedTensor with a matrix."""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        tensor = TrackedTensor(
            data=data,
            name="my_matrix",
            kind=TensorKind.MATRIX,
            tags=frozenset(["test"]),
        )

        assert tensor.name == "my_matrix"
        assert tensor.kind == TensorKind.MATRIX
        assert tensor.shape == (3, 2)
        assert tensor.tags == frozenset(["test"])

    def test_with_tags(self):
        """Can add tags to a tensor."""
        tensor = TrackedTensor(
            data=np.eye(2),
            name="identity",
            kind=TensorKind.MATRIX,
        )
        tagged = tensor.with_tags("symmetric", "psd")

        assert "symmetric" in tagged.tags
        assert "psd" in tagged.tags
        assert tensor.id == tagged.id  # Same tensor, same ID


class TestTensorSummary:
    """Tests for compute_summary()."""

    def test_vector_summary(self):
        """compute_summary() works for vectors."""
        tensor = TrackedTensor(
            data=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            name="test_vec",
            kind=TensorKind.VECTOR,
        )
        summary = compute_summary(tensor)

        assert summary.name == "test_vec"
        assert summary.kind == "vector"
        assert summary.shape == (5,)
        assert summary.stats["mean"] == 3.0
        assert summary.stats["min"] == 1.0
        assert summary.stats["max"] == 5.0
        assert "vector_stem" in summary.recommended_views

    def test_matrix_summary(self):
        """compute_summary() computes correct stats for matrices."""
        # Create a simple 2x2 symmetric PSD matrix
        data = np.array([[4.0, 2.0], [2.0, 3.0]])
        tensor = TrackedTensor(
            data=data,
            name="test_mat",
            kind=TensorKind.MATRIX,
        )
        summary = compute_summary(tensor)

        assert summary.shape == (2, 2)
        assert summary.stats["is_symmetric"] is True
        assert summary.stats["is_positive_definite"] is True
        assert summary.stats["rank"] == 2
        assert "heatmap" in summary.recommended_views
        assert "ellipse_2d" in summary.recommended_views

    def test_condition_number(self):
        """compute_summary() computes condition number correctly."""
        # Diagonal matrix with known condition number
        data = np.diag([10.0, 1.0])
        tensor = TrackedTensor(
            data=data,
            name="diag",
            kind=TensorKind.MATRIX,
        )
        summary = compute_summary(tensor)

        assert abs(summary.stats["condition_number"] - 10.0) < 1e-10


class TestOperator:
    """Tests for Operator ABC."""

    def test_define_operator(self):
        """Can define a custom Operator subclass."""

        class ScaleOperator(Operator):
            """Multiplies input by a scalar."""

            def __init__(self, scale: float = 2.0):
                self._scale = scale

            @property
            def name(self) -> str:
                return "scale"

            @property
            def input_specs(self) -> dict[str, TensorSpec]:
                return {
                    "input": TensorSpec(name="input"),
                }

            @property
            def output_specs(self) -> dict[str, TensorSpec]:
                return {
                    "output": TensorSpec(name="output"),
                }

            def forward(
                self, inputs: dict[str, TrackedTensor]
            ) -> dict[str, TrackedTensor]:
                inp = inputs["input"]
                return {
                    "output": TrackedTensor(
                        data=inp.data * self._scale,
                        name=f"scaled_{inp.name}",
                        kind=inp.kind,
                    )
                }

        op = ScaleOperator(scale=3.0)
        assert op.name == "scale"

        tensor = TrackedTensor(
            data=np.array([1.0, 2.0]),
            name="x",
            kind=TensorKind.VECTOR,
        )
        result = op({"input": tensor})

        assert "output" in result
        assert np.array_equal(result["output"].data, np.array([3.0, 6.0]))

    def test_tensor_spec_validation(self):
        """TensorSpec validation works correctly."""
        spec = TensorSpec(
            name="matrix_input",
            kind=TensorKind.MATRIX,
            shape=(-1, 2),  # Any rows, 2 columns
        )

        # Valid tensor
        valid = TrackedTensor(
            data=np.zeros((5, 2)),
            name="valid",
            kind=TensorKind.MATRIX,
        )
        is_valid, error = spec.validate(valid)
        assert is_valid

        # Wrong kind
        wrong_kind = TrackedTensor(
            data=np.zeros((5, 2)),
            name="wrong_kind",
            kind=TensorKind.IMAGE,
        )
        is_valid, error = spec.validate(wrong_kind)
        assert not is_valid
        assert "kind" in error

        # Wrong shape
        wrong_shape = TrackedTensor(
            data=np.zeros((5, 3)),
            name="wrong_shape",
            kind=TensorKind.MATRIX,
        )
        is_valid, error = spec.validate(wrong_shape)
        assert not is_valid
        assert "dimension" in error


class TestOperatorGraph:
    """Tests for OperatorGraph."""

    def test_build_simple_graph(self):
        """Can build a simple 2-node graph and execute it."""

        class AddOneOperator(Operator):
            @property
            def name(self) -> str:
                return "add_one"

            @property
            def input_specs(self) -> dict[str, TensorSpec]:
                return {"input": TensorSpec(name="input")}

            @property
            def output_specs(self) -> dict[str, TensorSpec]:
                return {"output": TensorSpec(name="output")}

            def forward(
                self, inputs: dict[str, TrackedTensor]
            ) -> dict[str, TrackedTensor]:
                inp = inputs["input"]
                return {
                    "output": TrackedTensor(
                        data=inp.data + 1,
                        name=f"{inp.name}_plus_1",
                        kind=inp.kind,
                    )
                }

        # Build graph: input -> add_one -> add_one -> output
        graph = OperatorGraph()
        graph.add_node(AddOneOperator(), "first")
        graph.add_node(AddOneOperator(), "second")

        graph.connect("_input", "x", "first", "input")
        graph.connect("first", "output", "second", "input")

        # Execute
        x = TrackedTensor(
            data=np.array([1.0, 2.0, 3.0]),
            name="x",
            kind=TensorKind.VECTOR,
        )
        results = graph.execute({"x": x})

        # Check results
        assert "first.output" in results
        assert "second.output" in results
        assert np.array_equal(results["first.output"].data, np.array([2.0, 3.0, 4.0]))
        assert np.array_equal(results["second.output"].data, np.array([3.0, 4.0, 5.0]))

    def test_graph_with_multiple_inputs(self):
        """Can build a graph with an operator that takes multiple inputs."""

        class AddOperator(Operator):
            @property
            def name(self) -> str:
                return "add"

            @property
            def input_specs(self) -> dict[str, TensorSpec]:
                return {
                    "a": TensorSpec(name="a"),
                    "b": TensorSpec(name="b"),
                }

            @property
            def output_specs(self) -> dict[str, TensorSpec]:
                return {"sum": TensorSpec(name="sum")}

            def forward(
                self, inputs: dict[str, TrackedTensor]
            ) -> dict[str, TrackedTensor]:
                a, b = inputs["a"], inputs["b"]
                return {
                    "sum": TrackedTensor(
                        data=a.data + b.data,
                        name=f"{a.name}+{b.name}",
                        kind=a.kind,
                    )
                }

        graph = OperatorGraph()
        graph.add_node(AddOperator(), "add")
        graph.connect("_input", "x", "add", "a")
        graph.connect("_input", "y", "add", "b")

        x = TrackedTensor(
            data=np.array([1.0, 2.0]),
            name="x",
            kind=TensorKind.VECTOR,
        )
        y = TrackedTensor(
            data=np.array([10.0, 20.0]),
            name="y",
            kind=TensorKind.VECTOR,
        )

        results = graph.execute({"x": x, "y": y})
        assert np.array_equal(results["add.sum"].data, np.array([11.0, 22.0]))

    def test_get_all_tensors(self):
        """get_all_tensors() returns all intermediate tensors."""

        class IdentityOperator(Operator):
            @property
            def name(self) -> str:
                return "identity"

            @property
            def input_specs(self) -> dict[str, TensorSpec]:
                return {"input": TensorSpec(name="input")}

            @property
            def output_specs(self) -> dict[str, TensorSpec]:
                return {"output": TensorSpec(name="output")}

            def forward(
                self, inputs: dict[str, TrackedTensor]
            ) -> dict[str, TrackedTensor]:
                return {"output": inputs["input"]}

        graph = OperatorGraph()
        graph.add_node(IdentityOperator(), "node1")
        graph.connect("_input", "x", "node1", "input")

        x = TrackedTensor(
            data=np.array([1.0]),
            name="x",
            kind=TensorKind.VECTOR,
        )
        graph.execute({"x": x})

        all_tensors = graph.get_all_tensors()
        assert "_input.x" in all_tensors
        assert "node1.output" in all_tensors


class TestOperatorRegistry:
    """Tests for OperatorRegistry."""

    def test_register_and_get(self):
        """Can register and retrieve operators."""
        OperatorRegistry.clear()

        @register_operator
        class TestOperator(Operator):
            @property
            def name(self) -> str:
                return "test_op"

            @property
            def input_specs(self) -> dict[str, TensorSpec]:
                return {}

            @property
            def output_specs(self) -> dict[str, TensorSpec]:
                return {}

            def forward(
                self, inputs: dict[str, TrackedTensor]
            ) -> dict[str, TrackedTensor]:
                return {}

        retrieved = OperatorRegistry.get("test_op")
        assert retrieved is TestOperator

        all_ops = OperatorRegistry.get_all()
        assert "test_op" in all_ops
