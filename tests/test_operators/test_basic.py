"""Tests for basic linear algebra operators."""

import numpy as np
import pytest

from tensorscope.core.tensor import TrackedTensor, TensorKind
from tensorscope.operators import (
    MatMul,
    Transpose,
    Norm,
    Add,
    Subtract,
    Scale,
    SVD,
    Eigendecomposition,
    QR,
    Cholesky,
    LeastSquares,
    LinearSolve,
    NormalEquations,
    Inverse,
    PseudoInverse,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def matrix_3x2():
    """A 3x2 matrix for testing."""
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    return TrackedTensor(data=data, name="A", kind=TensorKind.MATRIX)


@pytest.fixture
def matrix_2x3():
    """A 2x3 matrix for testing."""
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    return TrackedTensor(data=data, name="B", kind=TensorKind.MATRIX)


@pytest.fixture
def matrix_2x2():
    """A 2x2 matrix for testing."""
    data = np.array([[4.0, 2.0], [2.0, 3.0]])  # Symmetric positive definite
    return TrackedTensor(data=data, name="M", kind=TensorKind.MATRIX)


@pytest.fixture
def matrix_3x3_symmetric():
    """A 3x3 symmetric positive definite matrix."""
    # Create SPD matrix: A^T A + I
    A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    data = A @ A.T + np.eye(3)
    return TrackedTensor(data=data, name="S", kind=TensorKind.MATRIX)


@pytest.fixture
def vector_3():
    """A 3-element vector."""
    data = np.array([1.0, 2.0, 3.0])
    return TrackedTensor(data=data, name="v", kind=TensorKind.VECTOR)


@pytest.fixture
def vector_2():
    """A 2-element vector."""
    data = np.array([1.0, 2.0])
    return TrackedTensor(data=data, name="u", kind=TensorKind.VECTOR)


# ============================================================================
# Basic Operators
# ============================================================================


class TestMatMul:
    """Tests for MatMul operator."""

    def test_matrix_matrix(self, matrix_3x2, matrix_2x3):
        """Test matrix-matrix multiplication."""
        op = MatMul()
        result = op({"A": matrix_3x2, "B": matrix_2x3})

        assert "C" in result
        C = result["C"]

        # Verify shape
        assert C.shape == (3, 3)

        # Verify values match numpy
        expected = matrix_3x2.data @ matrix_2x3.data
        np.testing.assert_allclose(C.data, expected)

    def test_matrix_vector(self, matrix_3x2, vector_2):
        """Test matrix-vector multiplication."""
        op = MatMul()
        result = op({"A": matrix_3x2, "B": vector_2})

        C = result["C"]
        assert C.shape == (3,)
        assert C.kind == TensorKind.VECTOR

        expected = matrix_3x2.data @ vector_2.data
        np.testing.assert_allclose(C.data, expected)

    def test_symmetric_output_tagged(self):
        """Test that symmetric outputs get proper tags."""
        # A^T A produces symmetric PSD matrix
        A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        At = TrackedTensor(data=A.T, name="At", kind=TensorKind.MATRIX)
        A_tensor = TrackedTensor(data=A, name="A", kind=TensorKind.MATRIX)

        op = MatMul()
        result = op({"A": At, "B": A_tensor})

        C = result["C"]
        assert "symmetric" in C.tags
        assert "psd" in C.tags


class TestTranspose:
    """Tests for Transpose operator."""

    def test_transpose(self, matrix_3x2):
        """Test matrix transpose."""
        op = Transpose()
        result = op({"A": matrix_3x2})

        At = result["At"]
        assert At.shape == (2, 3)
        np.testing.assert_allclose(At.data, matrix_3x2.data.T)

    def test_transpose_preserves_tags(self):
        """Test that transpose preserves relevant tags."""
        data = np.array([[1.0, 0.0], [0.0, 1.0]])
        A = TrackedTensor(
            data=data,
            name="I",
            kind=TensorKind.MATRIX,
            tags=frozenset({"symmetric", "orthogonal"}),
        )

        op = Transpose()
        result = op({"A": A})

        assert "symmetric" in result["At"].tags
        assert "orthogonal" in result["At"].tags


class TestNorm:
    """Tests for Norm operator."""

    def test_frobenius_norm(self, matrix_3x2):
        """Test Frobenius norm."""
        op = Norm(ord="fro")
        result = op({"A": matrix_3x2})

        norm = result["norm"]
        expected = np.linalg.norm(matrix_3x2.data, ord="fro")
        np.testing.assert_allclose(norm.data, expected)
        assert "scalar" in norm.tags
        assert "non_negative" in norm.tags

    def test_l2_norm_vector(self, vector_3):
        """Test L2 norm of vector."""
        op = Norm(ord="l2")
        result = op({"A": vector_3})

        expected = np.linalg.norm(vector_3.data, ord=2)
        np.testing.assert_allclose(result["norm"].data, expected)


class TestAdd:
    """Tests for Add operator."""

    def test_add_matrices(self, matrix_3x2):
        """Test matrix addition."""
        op = Add()
        result = op({"A": matrix_3x2, "B": matrix_3x2})

        C = result["C"]
        expected = matrix_3x2.data + matrix_3x2.data
        np.testing.assert_allclose(C.data, expected)

    def test_add_vectors(self, vector_3):
        """Test vector addition."""
        op = Add()
        result = op({"A": vector_3, "B": vector_3})

        C = result["C"]
        assert C.kind == TensorKind.VECTOR
        np.testing.assert_allclose(C.data, vector_3.data * 2)


class TestSubtract:
    """Tests for Subtract operator."""

    def test_subtract(self, matrix_3x2):
        """Test matrix subtraction."""
        op = Subtract()
        result = op({"A": matrix_3x2, "B": matrix_3x2})

        C = result["C"]
        np.testing.assert_allclose(C.data, np.zeros_like(matrix_3x2.data))


class TestScale:
    """Tests for Scale operator."""

    def test_scale(self, matrix_3x2):
        """Test scalar multiplication."""
        op = Scale(alpha=2.5)
        result = op({"A": matrix_3x2})

        B = result["B"]
        np.testing.assert_allclose(B.data, 2.5 * matrix_3x2.data)


# ============================================================================
# Decomposition Operators
# ============================================================================


class TestSVD:
    """Tests for SVD operator."""

    def test_svd_shapes(self, matrix_3x2):
        """Test SVD output shapes."""
        op = SVD()
        result = op({"A": matrix_3x2})

        U = result["U"]
        S = result["S"]
        Vt = result["Vt"]

        # Reduced SVD: U is m x k, S is k, Vt is k x n where k = min(m, n)
        assert U.shape == (3, 2)
        assert S.shape == (2,)
        assert Vt.shape == (2, 2)

    def test_svd_reconstruction(self, matrix_3x2):
        """Test that U @ diag(S) @ Vt reconstructs A."""
        op = SVD()
        result = op({"A": matrix_3x2})

        U = result["U"].data
        S = result["S"].data
        Vt = result["Vt"].data

        reconstructed = U @ np.diag(S) @ Vt
        np.testing.assert_allclose(reconstructed, matrix_3x2.data, atol=1e-10)

    def test_svd_tags(self, matrix_3x2):
        """Test that SVD outputs have correct tags."""
        op = SVD()
        result = op({"A": matrix_3x2})

        assert "orthogonal" in result["U"].tags
        assert "orthogonal" in result["Vt"].tags
        assert "singular_values" in result["S"].tags
        assert "sorted_descending" in result["S"].tags


class TestEigendecomposition:
    """Tests for Eigendecomposition operator."""

    def test_eigendecomposition(self, matrix_2x2):
        """Test eigendecomposition of symmetric matrix."""
        op = Eigendecomposition()
        result = op({"A": matrix_2x2})

        eigenvalues = result["eigenvalues"]
        eigenvectors = result["eigenvectors"]

        # Verify reconstruction: V @ diag(λ) @ V^T = A
        V = eigenvectors.data
        L = eigenvalues.data
        reconstructed = V @ np.diag(L) @ V.T

        np.testing.assert_allclose(reconstructed, matrix_2x2.data, atol=1e-10)

    def test_eigenvalues_sorted(self, matrix_2x2):
        """Test that eigenvalues are sorted ascending."""
        op = Eigendecomposition()
        result = op({"A": matrix_2x2})

        eigenvalues = result["eigenvalues"].data
        assert np.all(eigenvalues[:-1] <= eigenvalues[1:])
        assert "sorted_ascending" in result["eigenvalues"].tags


class TestQR:
    """Tests for QR decomposition operator."""

    def test_qr_reconstruction(self, matrix_3x2):
        """Test QR reconstruction."""
        op = QR()
        result = op({"A": matrix_3x2})

        Q = result["Q"].data
        R = result["R"].data

        reconstructed = Q @ R
        np.testing.assert_allclose(reconstructed, matrix_3x2.data, atol=1e-10)

    def test_q_orthogonal(self, matrix_3x2):
        """Test that Q is orthogonal."""
        op = QR()
        result = op({"A": matrix_3x2})

        Q = result["Q"].data
        QtQ = Q.T @ Q
        np.testing.assert_allclose(QtQ, np.eye(Q.shape[1]), atol=1e-10)
        assert "orthogonal" in result["Q"].tags


class TestCholesky:
    """Tests for Cholesky decomposition."""

    def test_cholesky(self, matrix_2x2):
        """Test Cholesky decomposition."""
        op = Cholesky()
        result = op({"A": matrix_2x2})

        L = result["L"].data
        reconstructed = L @ L.T
        np.testing.assert_allclose(reconstructed, matrix_2x2.data, atol=1e-10)
        assert "lower_triangular" in result["L"].tags


# ============================================================================
# Solver Operators
# ============================================================================


class TestLeastSquares:
    """Tests for LeastSquares operator."""

    def test_overdetermined(self, matrix_3x2, vector_3):
        """Test least squares for overdetermined system."""
        op = LeastSquares()
        result = op({"A": matrix_3x2, "b": vector_3})

        x = result["x"]
        residual = result["residual"]

        # Verify solution matches numpy
        expected_x, _, _, _ = np.linalg.lstsq(
            matrix_3x2.data, vector_3.data, rcond=None
        )
        np.testing.assert_allclose(x.data, expected_x, atol=1e-10)

        # Verify residual
        expected_residual = vector_3.data - matrix_3x2.data @ x.data
        np.testing.assert_allclose(residual.data, expected_residual, atol=1e-10)

    def test_exact_solution(self, matrix_2x2, vector_2):
        """Test least squares for square system (exact solution)."""
        op = LeastSquares()
        result = op({"A": matrix_2x2, "b": vector_2})

        x = result["x"]
        residual_norm = result["residual_norm"]

        # Verify Ax ≈ b
        np.testing.assert_allclose(
            matrix_2x2.data @ x.data, vector_2.data, atol=1e-10
        )
        # Residual should be nearly zero
        assert residual_norm.data < 1e-10


class TestLinearSolve:
    """Tests for LinearSolve operator."""

    def test_solve(self, matrix_2x2, vector_2):
        """Test direct linear solve."""
        op = LinearSolve()
        result = op({"A": matrix_2x2, "b": vector_2})

        x = result["x"]
        expected = np.linalg.solve(matrix_2x2.data, vector_2.data)
        np.testing.assert_allclose(x.data, expected, atol=1e-10)

    def test_non_square_raises(self, matrix_3x2, vector_3):
        """Test that non-square matrix raises error."""
        op = LinearSolve()
        with pytest.raises(ValueError, match="square"):
            op({"A": matrix_3x2, "b": vector_3})


class TestNormalEquations:
    """Tests for NormalEquations operator."""

    def test_normal_equations(self, matrix_3x2, vector_3):
        """Test normal equations solver."""
        op = NormalEquations()
        result = op({"A": matrix_3x2, "b": vector_3})

        x = result["x"]
        AtA = result["AtA"]
        Atb = result["Atb"]

        # Verify AtA is symmetric PSD
        assert "symmetric" in AtA.tags
        assert "psd" in AtA.tags

        # Verify solution
        expected_x, _, _, _ = np.linalg.lstsq(
            matrix_3x2.data, vector_3.data, rcond=None
        )
        np.testing.assert_allclose(x.data, expected_x, atol=1e-10)

        # Verify AtA and Atb
        np.testing.assert_allclose(AtA.data, matrix_3x2.data.T @ matrix_3x2.data)
        np.testing.assert_allclose(Atb.data, matrix_3x2.data.T @ vector_3.data)


class TestInverse:
    """Tests for Inverse operator."""

    def test_inverse(self, matrix_2x2):
        """Test matrix inverse."""
        op = Inverse()
        result = op({"A": matrix_2x2})

        A_inv = result["A_inv"]

        # Verify A @ A^{-1} = I
        product = matrix_2x2.data @ A_inv.data
        np.testing.assert_allclose(product, np.eye(2), atol=1e-10)

    def test_non_square_raises(self, matrix_3x2):
        """Test that non-square matrix raises error."""
        op = Inverse()
        with pytest.raises(ValueError, match="square"):
            op({"A": matrix_3x2})


class TestPseudoInverse:
    """Tests for PseudoInverse operator."""

    def test_pseudoinverse_overdetermined(self, matrix_3x2):
        """Test pseudoinverse of tall matrix."""
        op = PseudoInverse()
        result = op({"A": matrix_3x2})

        A_pinv = result["A_pinv"]
        assert A_pinv.shape == (2, 3)

        # Verify A^+ @ A = I (left inverse property for full column rank)
        product = A_pinv.data @ matrix_3x2.data
        np.testing.assert_allclose(product, np.eye(2), atol=1e-10)

    def test_pseudoinverse_square(self, matrix_2x2):
        """Test pseudoinverse of square invertible matrix equals inverse."""
        op = PseudoInverse()
        result = op({"A": matrix_2x2})

        A_pinv = result["A_pinv"]
        expected = np.linalg.inv(matrix_2x2.data)
        np.testing.assert_allclose(A_pinv.data, expected, atol=1e-10)
