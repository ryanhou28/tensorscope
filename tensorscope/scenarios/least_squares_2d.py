"""Least Squares 2D scenario for Tensorscope.

This scenario demonstrates the geometry of least squares projection
using a simple overdetermined 2D system (3 equations, 2 unknowns).

Key concepts visualized:
- Column space of A (a 2D plane in 3D space)
- Projection of b onto the column space
- Residual vector (orthogonal to column space)
- Normal equations: A^T A x = A^T b
- Effect of noise and conditioning on the solution
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.tensor import TrackedTensor, TensorKind
from ..core.graph import OperatorGraph
from ..core.scenario import Scenario
from ..operators.basic import MatMul, Transpose, Subtract
from ..operators.solvers import LinearSolve


def create_least_squares_2d_scenario() -> Scenario:
    """Create and configure the Least Squares 2D scenario.

    This scenario solves the overdetermined system Ax = b in the least
    squares sense, where A is 3x2 and b is a 3-vector.

    The solution is computed via normal equations:
        x* = (A^T A)^{-1} A^T b

    Parameters:
        noise_level: Amount of noise added to b (0.0 to 1.0)
        condition_number: Condition number of A (1 to 100)

    Probed tensors:
        - A: The design matrix
        - AtA: Normal matrix (shows conditioning)
        - x: Least squares solution
        - projection: A @ x (point in column space closest to b)
        - residual: b - A @ x (error vector)

    Returns:
        A configured Scenario ready to run.
    """
    scenario = Scenario(
        name="Least Squares 2D",
        description=(
            "Demonstrates least squares projection for an overdetermined "
            "2D system. Shows how noise and conditioning affect the solution."
        ),
        id="least_squares_2d",
    )

    # Define parameters
    scenario.param(
        "noise_level",
        display_name="Noise Level",
        min_val=0.0,
        max_val=1.0,
        default=0.1,
        step=0.01,
        description="Amount of noise added to the observation vector b",
    )

    scenario.param(
        "condition_number",
        display_name="Condition Number",
        min_val=1.0,
        max_val=100.0,
        default=10.0,
        step=1.0,
        description="Condition number of the design matrix A",
    )

    scenario.param(
        "seed",
        display_name="Random Seed",
        min_val=0,
        max_val=1000,
        default=42,
        step=1,
        description="Random seed for reproducible results",
    )

    # Build operator graph
    graph = OperatorGraph()

    # Operators for A^T A
    transpose_a = Transpose()
    graph.add_node(transpose_a, "transpose_A")
    graph.connect("_input", "A", "transpose_A", "A")

    # AtA = A^T @ A
    matmul_ata = MatMul()
    graph.add_node(matmul_ata, "AtA")
    graph.connect("transpose_A", "At", "AtA", "A")
    graph.connect("_input", "A", "AtA", "B")

    # Atb = A^T @ b
    matmul_atb = MatMul()
    graph.add_node(matmul_atb, "Atb")
    graph.connect("transpose_A", "At", "Atb", "A")
    graph.connect("_input", "b", "Atb", "B")

    # x = solve(AtA, Atb)
    solve = LinearSolve()
    graph.add_node(solve, "solve")
    graph.connect("AtA", "C", "solve", "A")
    graph.connect("Atb", "C", "solve", "b")

    # projection = A @ x
    matmul_proj = MatMul()
    graph.add_node(matmul_proj, "projection")
    graph.connect("_input", "A", "projection", "A")
    graph.connect("solve", "x", "projection", "B")

    # residual = b - projection
    subtract = Subtract()
    graph.add_node(subtract, "residual")
    graph.connect("_input", "b", "residual", "A")
    graph.connect("projection", "C", "residual", "B")

    scenario.set_graph(graph)

    # Set up input generator
    def generate_inputs(params: dict[str, Any]) -> dict[str, TrackedTensor]:
        """Generate input tensors based on parameters."""
        noise_level = params["noise_level"]
        condition_number = params["condition_number"]
        seed = int(params["seed"])

        rng = np.random.default_rng(seed)

        # Create A with specified condition number
        # Start with a well-conditioned matrix
        A_base = rng.standard_normal((3, 2))

        # Use SVD to control condition number
        U, s, Vt = np.linalg.svd(A_base, full_matrices=False)

        # Set singular values to achieve desired condition number
        # s[0] / s[1] = condition_number
        s_new = np.array([condition_number, 1.0])
        A = U @ np.diag(s_new) @ Vt

        # Create "true" solution
        x_true = np.array([1.0, 2.0])

        # Create b = A @ x_true + noise
        b_clean = A @ x_true
        noise = noise_level * rng.standard_normal(3)
        b = b_clean + noise

        return {
            "A": TrackedTensor(
                data=A,
                name="A",
                kind=TensorKind.MATRIX,
                tags=frozenset({"design_matrix"}),
            ),
            "b": TrackedTensor(
                data=b,
                name="b",
                kind=TensorKind.VECTOR,
                tags=frozenset({"observation"}),
            ),
        }

    scenario.set_input_generator(generate_inputs)

    # Mark tensors for inspection
    scenario.probe(
        "_input.A",
        display_name="A (Design Matrix)",
        description="The 3x2 design matrix defining the overdetermined system",
    )
    scenario.probe(
        "_input.b",
        display_name="b (Observation)",
        description="The observation vector (with optional noise)",
    )
    scenario.probe(
        "AtA.C",
        display_name="A^T A (Normal Matrix)",
        description="The normal matrix - its conditioning affects solution stability",
    )
    scenario.probe(
        "Atb.C",
        display_name="A^T b",
        description="The right-hand side of the normal equations",
    )
    scenario.probe(
        "solve.x",
        display_name="x* (Solution)",
        description="The least squares solution vector",
    )
    scenario.probe(
        "projection.C",
        display_name="Ax* (Projection)",
        description="The projection of b onto the column space of A",
    )
    scenario.probe(
        "residual.C",
        display_name="r (Residual)",
        description="The residual vector b - Ax*, orthogonal to column space",
    )

    return scenario


# Pre-built scenario instance for easy import
least_squares_2d = create_least_squares_2d_scenario()
