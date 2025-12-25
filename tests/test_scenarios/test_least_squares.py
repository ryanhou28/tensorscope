"""Tests for the Least Squares 2D scenario."""

import numpy as np
import pytest

from tensorscope.core import Scenario, Parameter, ParameterType
from tensorscope.scenarios import least_squares_2d, create_least_squares_2d_scenario


class TestLeastSquares2DScenario:
    """Test suite for the Least Squares 2D scenario."""

    def test_scenario_creation(self):
        """Test that the scenario is created correctly."""
        scenario = create_least_squares_2d_scenario()

        assert scenario.id == "least_squares_2d"
        assert scenario.name == "Least Squares 2D"
        assert len(scenario.description) > 0

    def test_parameters_defined(self):
        """Test that all expected parameters are defined."""
        scenario = least_squares_2d

        params = scenario.parameters
        assert "noise_level" in params
        assert "condition_number" in params
        assert "seed" in params

        # Check noise_level parameter
        noise_param = params["noise_level"]
        assert noise_param.param_type == ParameterType.CONTINUOUS
        assert noise_param.min_val == 0.0
        assert noise_param.max_val == 1.0
        assert noise_param.default == 0.1

        # Check condition_number parameter
        cond_param = params["condition_number"]
        assert cond_param.param_type == ParameterType.CONTINUOUS
        assert cond_param.min_val == 1.0
        assert cond_param.max_val == 100.0
        assert cond_param.default == 10.0

    def test_probes_defined(self):
        """Test that probe points are defined."""
        scenario = least_squares_2d

        probes = scenario.probes
        assert len(probes) >= 5  # At minimum: A, b, AtA, x, residual

        probe_keys = [p.tensor_key for p in probes]
        assert "_input.A" in probe_keys
        assert "_input.b" in probe_keys
        assert "AtA.C" in probe_keys
        assert "solve.x" in probe_keys
        assert "residual.C" in probe_keys

    def test_graph_defined(self):
        """Test that the operator graph is properly configured."""
        scenario = least_squares_2d

        graph = scenario.graph
        assert graph is not None

        nodes = graph.get_nodes()
        assert len(nodes) > 0

        # Check key nodes exist
        node_names = list(nodes.keys())
        assert "transpose_A" in node_names
        assert "AtA" in node_names
        assert "solve" in node_names
        assert "projection" in node_names
        assert "residual" in node_names

    def test_run_with_defaults(self):
        """Test running the scenario with default parameters."""
        scenario = create_least_squares_2d_scenario()

        results = scenario.run()

        assert len(results) > 0

        # Check key outputs exist
        assert "_input.A" in results
        assert "_input.b" in results
        assert "AtA.C" in results
        assert "solve.x" in results
        assert "projection.C" in results
        assert "residual.C" in results

    def test_output_shapes(self):
        """Test that output tensors have correct shapes."""
        scenario = create_least_squares_2d_scenario()
        results = scenario.run()

        # A is 3x2
        A = results["_input.A"]
        assert A.shape == (3, 2)

        # b is 3-vector
        b = results["_input.b"]
        assert b.shape == (3,)

        # AtA is 2x2
        AtA = results["AtA.C"]
        assert AtA.shape == (2, 2)

        # x is 2-vector
        x = results["solve.x"]
        assert x.shape == (2,)

        # projection is 3-vector
        projection = results["projection.C"]
        assert projection.shape == (3,)

        # residual is 3-vector
        residual = results["residual.C"]
        assert residual.shape == (3,)

    def test_mathematical_correctness(self):
        """Test that the least squares solution is mathematically correct."""
        scenario = create_least_squares_2d_scenario()
        results = scenario.run({"noise_level": 0.0})  # No noise for clean test

        A = results["_input.A"].data
        b = results["_input.b"].data
        x = results["solve.x"].data
        projection = results["projection.C"].data
        residual = results["residual.C"].data

        # Test: projection = A @ x
        expected_projection = A @ x
        np.testing.assert_allclose(projection, expected_projection, rtol=1e-10)

        # Test: residual = b - projection
        expected_residual = b - projection
        np.testing.assert_allclose(residual, expected_residual, rtol=1e-10)

        # Test: residual is orthogonal to column space of A
        # (A^T @ residual should be ~0)
        orthogonality_error = A.T @ residual
        np.testing.assert_allclose(orthogonality_error, np.zeros(2), atol=1e-10)

        # Test: x matches numpy's lstsq solution
        x_numpy, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        np.testing.assert_allclose(x, x_numpy, rtol=1e-10)

    def test_ata_is_symmetric_psd(self):
        """Test that A^T A is symmetric and positive semi-definite."""
        scenario = create_least_squares_2d_scenario()
        results = scenario.run()

        AtA = results["AtA.C"].data

        # Check symmetry
        np.testing.assert_allclose(AtA, AtA.T, rtol=1e-10)

        # Check positive semi-definiteness (eigenvalues >= 0)
        eigenvalues = np.linalg.eigvalsh(AtA)
        assert np.all(eigenvalues >= -1e-10)

    def test_noise_affects_results(self):
        """Test that changing noise_level affects the results."""
        scenario = create_least_squares_2d_scenario()

        # Run with no noise
        results_clean = scenario.run({"noise_level": 0.0, "seed": 42})
        x_clean = results_clean["solve.x"].data
        residual_clean = results_clean["residual.C"].data

        # Run with high noise
        scenario2 = create_least_squares_2d_scenario()
        results_noisy = scenario2.run({"noise_level": 0.5, "seed": 42})
        x_noisy = results_noisy["solve.x"].data
        residual_noisy = results_noisy["residual.C"].data

        # Solutions should be different
        assert not np.allclose(x_clean, x_noisy)

        # Residual norm should be larger with noise
        residual_norm_clean = np.linalg.norm(residual_clean)
        residual_norm_noisy = np.linalg.norm(residual_noisy)
        # Note: This isn't always true due to random noise, but usually is
        # We just check they're different
        assert residual_norm_clean != residual_norm_noisy

    def test_condition_number_affects_matrix(self):
        """Test that condition_number parameter affects A^T A."""
        scenario1 = create_least_squares_2d_scenario()
        results1 = scenario1.run({"condition_number": 1.0, "seed": 42})
        AtA1 = results1["AtA.C"].data

        scenario2 = create_least_squares_2d_scenario()
        results2 = scenario2.run({"condition_number": 50.0, "seed": 42})
        AtA2 = results2["AtA.C"].data

        # Compute actual condition numbers
        cond1 = np.linalg.cond(AtA1)
        cond2 = np.linalg.cond(AtA2)

        # Higher condition_number parameter should lead to worse conditioning
        # (AtA condition number is roughly squared of A's condition number)
        assert cond2 > cond1

    def test_seed_reproducibility(self):
        """Test that same seed produces identical results."""
        scenario1 = create_least_squares_2d_scenario()
        results1 = scenario1.run({"seed": 123})

        scenario2 = create_least_squares_2d_scenario()
        results2 = scenario2.run({"seed": 123})

        # All tensors should be identical
        for key in results1:
            np.testing.assert_array_equal(
                results1[key].data,
                results2[key].data,
                err_msg=f"Mismatch in {key}",
            )

    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        scenario1 = create_least_squares_2d_scenario()
        results1 = scenario1.run({"seed": 1})

        scenario2 = create_least_squares_2d_scenario()
        results2 = scenario2.run({"seed": 2})

        # At least some tensors should differ
        A1 = results1["_input.A"].data
        A2 = results2["_input.A"].data
        assert not np.allclose(A1, A2)

    def test_get_probed_tensors(self):
        """Test getting only probed tensors."""
        scenario = create_least_squares_2d_scenario()
        scenario.run()

        probed = scenario.get_probed_tensors()

        # Should have entries for each probe's display_name
        assert "A (Design Matrix)" in probed
        assert "x* (Solution)" in probed
        assert "r (Residual)" in probed

    def test_get_probed_summaries(self):
        """Test getting summaries of probed tensors."""
        scenario = create_least_squares_2d_scenario()
        scenario.run()

        summaries = scenario.get_probed_summaries()

        assert len(summaries) > 0

        # Check a summary has expected fields
        for name, summary in summaries.items():
            assert summary.id is not None
            assert summary.name is not None
            assert summary.shape is not None
            assert summary.dtype is not None
            assert "norm" in summary.stats

    def test_to_dict_serialization(self):
        """Test scenario serialization for API."""
        scenario = least_squares_2d
        data = scenario.to_dict()

        assert data["id"] == "least_squares_2d"
        assert data["name"] == "Least Squares 2D"
        assert "parameters" in data
        assert "probes" in data
        assert "graph" in data

        # Check parameter serialization
        params = {p["name"]: p for p in data["parameters"]}
        assert "noise_level" in params
        assert params["noise_level"]["type"] == "continuous"
        assert params["noise_level"]["min"] == 0.0
        assert params["noise_level"]["max"] == 1.0

    def test_parameter_validation(self):
        """Test that invalid parameters are rejected."""
        scenario = create_least_squares_2d_scenario()

        # noise_level out of range
        is_valid, errors = scenario.validate_params({"noise_level": 2.0})
        assert not is_valid
        assert len(errors) > 0

        # Unknown parameter
        is_valid, errors = scenario.validate_params({"unknown_param": 1.0})
        assert not is_valid

    def test_run_raises_on_invalid_params(self):
        """Test that run() raises ValueError for invalid parameters."""
        scenario = create_least_squares_2d_scenario()

        with pytest.raises(ValueError):
            scenario.run({"noise_level": -1.0})  # Below minimum


class TestScenarioClass:
    """Test the Scenario base class functionality."""

    def test_scenario_without_graph_raises(self):
        """Test that running without a graph raises an error."""
        scenario = Scenario(name="test")
        scenario.set_input_generator(lambda p: {})

        with pytest.raises(ValueError, match="graph not set"):
            scenario.run()

    def test_scenario_without_generator_raises(self):
        """Test that running without an input generator raises an error."""
        from tensorscope.core import OperatorGraph

        scenario = Scenario(name="test")
        scenario.set_graph(OperatorGraph())

        with pytest.raises(ValueError, match="Input generator not set"):
            scenario.run()

    def test_parameter_to_dict(self):
        """Test Parameter serialization."""
        param = Parameter(
            name="test_param",
            display_name="Test Parameter",
            param_type=ParameterType.CONTINUOUS,
            default=0.5,
            min_val=0.0,
            max_val=1.0,
            step=0.1,
            description="A test parameter",
        )

        data = param.to_dict()
        assert data["name"] == "test_param"
        assert data["display_name"] == "Test Parameter"
        assert data["type"] == "continuous"
        assert data["default"] == 0.5
        assert data["min"] == 0.0
        assert data["max"] == 1.0
        assert data["step"] == 0.1

    def test_discrete_parameter(self):
        """Test discrete parameter creation and validation."""
        scenario = Scenario(name="test")
        scenario.param(
            "method",
            options=["svd", "normal", "qr"],
            default="svd",
        )

        params = scenario.parameters
        assert "method" in params
        assert params["method"].param_type == ParameterType.DISCRETE
        assert params["method"].options == ["svd", "normal", "qr"]

        # Valid value
        is_valid, _ = params["method"].validate("svd")
        assert is_valid

        # Invalid value
        is_valid, error = params["method"].validate("invalid")
        assert not is_valid
        assert "not in options" in error
