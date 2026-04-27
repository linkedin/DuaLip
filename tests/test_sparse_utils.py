import torch

from dualip.utils.sparse_utils import (
    apply_F_to_columns,
    hstack_csc,
    left_multiply_sparse,
    right_multiply_sparse,
    vstack_csc,
)


def test_vstack_csc():
    """Test vertical stacking using dense tensor reference."""
    # Create dense tensors
    A_dense = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
    B_dense = torch.tensor([[4.0, 5.0, 0.0], [0.0, 0.0, 6.0]])

    # Convert to sparse CSC
    A_sparse = A_dense.to_sparse_csc()
    B_sparse = B_dense.to_sparse_csc()

    # Test vstack_csc
    result_sparse = vstack_csc([A_sparse, B_sparse])

    # Compare with dense vstack
    expected_dense = torch.vstack([A_dense, B_dense])

    assert torch.allclose(result_sparse.to_dense(), expected_dense)
    assert result_sparse.layout == torch.sparse_csc


def test_hstack_csc():
    """Test horizontal stacking using dense tensor reference."""
    # Create dense tensors
    A_dense = torch.tensor([[1.0, 2.0], [3.0, 0.0]])
    B_dense = torch.tensor([[0.0, 4.0, 5.0], [6.0, 0.0, 7.0]])

    # Convert to sparse CSC
    A_sparse = A_dense.to_sparse_csc()
    B_sparse = B_dense.to_sparse_csc()

    # Test hstack_csc
    result_sparse = hstack_csc([A_sparse, B_sparse])

    # Compare with dense hstack
    expected_dense = torch.hstack([A_dense, B_dense])

    assert torch.allclose(result_sparse.to_dense(), expected_dense)
    assert result_sparse.layout == torch.sparse_csc


def test_combined_stacking():
    """Test combining both operations."""
    # Create four 2x2 dense matrices
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    C = torch.tensor([[9.0, 10.0], [11.0, 12.0]])
    D = torch.tensor([[13.0, 14.0], [15.0, 16.0]])

    # Convert to sparse
    A_sp, B_sp, C_sp, D_sp = [x.to_sparse_csc() for x in [A, B, C, D]]

    # Stack: [[A, B], [C, D]]
    top_sparse = hstack_csc([A_sp, B_sp])
    bottom_sparse = hstack_csc([C_sp, D_sp])
    result_sparse = vstack_csc([top_sparse, bottom_sparse])

    # Dense reference
    top_dense = torch.hstack([A, B])
    bottom_dense = torch.hstack([C, D])
    expected_dense = torch.vstack([top_dense, bottom_dense])

    assert torch.allclose(result_sparse.to_dense(), expected_dense)


def test_right_multiply_sparse():
    """Test right multiplication M @ diag(v) using dense tensor reference."""
    # Create a sparse matrix and diagonal vector
    M_dense = torch.tensor([[1.0, 0.0, 3.0], [0.0, 2.0, 0.0], [4.0, 0.0, 5.0]])
    v = torch.tensor([2.0, 3.0, 0.5])

    # Convert to sparse CSC
    M_sparse = M_dense.to_sparse_csc()

    # Test right_multiply_sparse
    result_sparse = right_multiply_sparse(M_sparse, v)

    # Compare with dense matrix multiplication
    expected_dense = M_dense @ torch.diag(v)

    assert torch.allclose(result_sparse.to_dense(), expected_dense)
    assert result_sparse.layout == torch.sparse_csc


class TestApplyFToColumns:
    """Tests for apply_F_to_columns with various bucket configurations."""

    @staticmethod
    def _apply_dense_columnwise(M_dense, F_batch):
        """Reference: apply F_batch to each column independently via dense ops."""
        result = M_dense.clone()
        for j in range(M_dense.shape[1]):
            col = M_dense[:, j]
            nonzero_mask = col != 0
            if nonzero_mask.any():
                block = col[nonzero_mask].unsqueeze(1)
                proj = F_batch(block).squeeze(1)
                result[:, j] = 0.0
                result[:, j][nonzero_mask] = proj
        return result

    def test_identity_function(self):
        """F = identity should return the same matrix."""
        M_dense = torch.tensor(
            [[1.0, 0.0, 3.0], [0.0, 2.0, 0.0], [4.0, 0.0, 5.0]]
        )
        M_sparse = M_dense.to_sparse_csc()
        buckets = [torch.arange(M_dense.shape[1])]

        result = apply_F_to_columns(M_sparse, lambda x: x, buckets)
        assert torch.allclose(result.to_dense(), M_dense)

    def test_scaling_function(self):
        """F = 2x should double all values."""
        M_dense = torch.tensor(
            [[1.0, 0.0, 3.0], [0.0, 2.0, 0.0], [4.0, 0.0, 5.0]]
        )
        M_sparse = M_dense.to_sparse_csc()
        buckets = [torch.arange(M_dense.shape[1])]

        result = apply_F_to_columns(M_sparse, lambda x: 2 * x, buckets)
        assert torch.allclose(result.to_dense(), 2 * M_dense)

    def test_multiple_buckets(self):
        """Splitting columns across multiple buckets should give the same result."""
        M_dense = torch.tensor(
            [
                [1.0, 0.0, 3.0, 0.0, 7.0],
                [0.0, 2.0, 0.0, 4.0, 0.0],
                [5.0, 0.0, 6.0, 0.0, 8.0],
            ]
        )
        M_sparse = M_dense.to_sparse_csc()
        f = lambda x: x * 0.5

        expected = M_dense * 0.5

        single = apply_F_to_columns(M_sparse, f, [torch.arange(5)])
        multi = apply_F_to_columns(
            M_sparse, f, [torch.tensor([0, 2, 4]), torch.tensor([1, 3])]
        )

        assert torch.allclose(single.to_dense(), expected)
        assert torch.allclose(multi.to_dense(), expected)

    def test_varying_column_lengths(self):
        """Columns with different numbers of nonzeros in the same bucket."""
        M_dense = torch.tensor(
            [
                [1.0, 0.0, 3.0],
                [2.0, 0.0, 0.0],
                [3.0, 4.0, 0.0],
                [4.0, 0.0, 0.0],
            ]
        )
        M_sparse = M_dense.to_sparse_csc()
        f = lambda x: x ** 2
        buckets = [torch.arange(3)]

        result = apply_F_to_columns(M_sparse, f, buckets)
        expected = self._apply_dense_columnwise(M_dense, f)
        assert torch.allclose(result.to_dense(), expected)

    def test_output_tensor(self):
        """Writing into a pre-allocated output tensor."""
        M_dense = torch.tensor(
            [[1.0, 0.0, 3.0], [0.0, 2.0, 0.0], [4.0, 0.0, 5.0]]
        )
        M_sparse = M_dense.to_sparse_csc()
        output = M_sparse.clone()
        buckets = [torch.arange(3)]

        apply_F_to_columns(M_sparse, lambda x: x * 3, buckets, output_tensor=output)
        assert torch.allclose(output.to_dense(), 3 * M_dense)

    def test_empty_bucket_skipped(self):
        """Empty buckets should be harmlessly skipped."""
        M_dense = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        M_sparse = M_dense.to_sparse_csc()
        buckets = [torch.tensor([], dtype=torch.long), torch.arange(2)]

        result = apply_F_to_columns(M_sparse, lambda x: -x, buckets)
        assert torch.allclose(result.to_dense(), -M_dense)

    def test_clamp_projection(self):
        """Column-wise clamp to [0, inf) (ReLU-like) preserves sparsity pattern."""
        M_dense = torch.tensor(
            [[1.0, 0.0, -3.0], [0.0, -2.0, 0.0], [-4.0, 0.0, 5.0]]
        )
        M_sparse = M_dense.to_sparse_csc()
        f = lambda x: x.clamp(min=0)
        buckets = [torch.arange(3)]

        result = apply_F_to_columns(M_sparse, f, buckets)
        expected = self._apply_dense_columnwise(M_dense, f)
        assert torch.allclose(result.to_dense(), expected)


def test_left_multiply_sparse():
    """Test left multiplication diag(v) @ M using dense tensor reference."""
    # Create a sparse matrix and diagonal vector
    M_dense = torch.tensor([[1.0, 0.0, 3.0], [0.0, 2.0, 0.0], [4.0, 0.0, 5.0]])
    v = torch.tensor([2.0, 3.0, 0.5])

    # Convert to sparse CSC
    M_sparse = M_dense.to_sparse_csc()

    # Test left_multiply_sparse
    result_sparse = left_multiply_sparse(v, M_sparse)

    # Compare with dense matrix multiplication
    expected_dense = torch.diag(v) @ M_dense

    assert torch.allclose(result_sparse.to_dense(), expected_dense)
    assert result_sparse.layout == torch.sparse_csc
