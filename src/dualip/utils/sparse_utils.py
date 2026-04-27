from typing import Callable, List, Optional

import torch


# Aida: Needed for primal objective computation, although is slows
def dot_product_csc(A: torch.Tensor, B: torch.Tensor):
    """
    Compute the dot product of two CSC‐format sparse vectors:
        A: Shape (m, n)
        B: Shape (m, n)
    Returns:
        scalar tensor equal to sum_{i, j} A_ij B_ij
    """
    assert A.layout == torch.sparse_csc and B.layout == torch.sparse_csc, "Inputs must both be CSC sparse tensors"

    m1, n1 = A.shape
    m2, n2 = B.shape

    assert m1 == m2 and n1 == n2, f"Expected shapes (m, n) and (m, n), got {A.shape} and {B.shape}"

    total = torch.dot(A.values(), B.values())
    return total


def elementwise_csc(A: torch.Tensor, B: torch.Tensor, op, output_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Element-wise apply `op` to two CSC sparse tensors A and B
    (they must have identical col_ptr & row_idx).

    If output_tensor is provided, it will be used to store the result and skip all
    sparsity pattern checks to speed up the operation.
    """
    if A.layout != torch.sparse_csc or B.layout != torch.sparse_csc:
        raise ValueError("Both A and B must be CSC-format sparse tensors")
    if output_tensor is None and not (
        torch.equal(A.ccol_indices(), B.ccol_indices()) and torch.equal(A.row_indices(), B.row_indices())
    ):
        raise ValueError("A and B must share the same sparsity pattern")

    ccol_ptr = A.ccol_indices()
    row_idx = A.row_indices()
    vals_A = A.values()
    vals_B = B.values()

    new_vals = op(vals_A, vals_B)  # e.g. sub or mul elementwise

    if output_tensor is None:
        return torch.sparse_csc_tensor(ccol_ptr, row_idx, new_vals, size=A.size())
    else:
        return output_tensor.values().copy_(new_vals)


def left_multiply_sparse(
    v: torch.Tensor, M: torch.Tensor, output_tensor: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Computes diag(v) @ M for a CSC-format sparse matrix M,
    preserving its sparsity.

    If output_tensor is provided, it will be used to store the result inplace rather
    than allocating a new tensor.

    Args:
        v:  1D dense tensor of length m.
        M:  torch.sparse_csc_tensor of shape (m, n).
        output_tensor: Optional tensor to store the result with sparsity pattern of M.
    Returns:
        A new torch.sparse_csc_tensor representing diag(v) @ M.
    """
    if M.layout != torch.sparse_csc:
        raise ValueError("Expected M to be a CSC-format sparse tensor")
    # Extract CSC storage
    ccol_ptr = M.ccol_indices()  # shape (n+1,)
    row_idx = M.row_indices()  # shape (nnz,)
    vals = M.values()  # shape (nnz,)

    # Scale each nonzero by the corresponding entry in v
    new_vals = vals * v[row_idx]

    # Rebuild and return
    if output_tensor is None:
        return torch.sparse_csc_tensor(ccol_ptr, row_idx, new_vals, size=M.size())
    else:
        return output_tensor.values().copy_(new_vals)


def right_multiply_sparse(
    M: torch.Tensor, v: torch.Tensor, output_tensor: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Computes M @ diag(v) for a CSC-format sparse matrix M,
    preserving its sparsity.

    If output_tensor is provided, it will be used to store the result inplace rather
    than allocating a new tensor.

    Args:
        M:  torch.sparse_csc_tensor of shape (m, n).
        v:  1D dense tensor of length n.
        output_tensor: Optional tensor to store the result with sparsity pattern of M.
    Returns:
        A new torch.sparse_csc_tensor representing M @ diag(v).
    """
    if M.layout != torch.sparse_csc:
        raise ValueError("Expected M to be a CSC-format sparse tensor")
    # Extract CSC storage
    ccol_ptr = M.ccol_indices()  # shape (n+1,)
    row_idx = M.row_indices()  # shape (nnz,)
    vals = M.values()  # shape (nnz,)

    # For M @ diag(v), we need to scale each column by the corresponding entry in v
    # We need to determine which column each nonzero belongs to
    n_cols = M.size(1)
    col_indices = torch.zeros_like(row_idx)

    # Build column indices for each nonzero
    for col in range(n_cols):
        start_idx = int(ccol_ptr[col].item())
        end_idx = int(ccol_ptr[col + 1].item())
        col_indices[start_idx:end_idx] = col

    # Scale each nonzero by the corresponding entry in v
    new_vals = vals * v[col_indices]

    # Rebuild and return
    if output_tensor is None:
        return torch.sparse_csc_tensor(ccol_ptr, row_idx, new_vals, size=M.size())
    else:
        return output_tensor.values().copy_(new_vals)


def apply_F_to_columns(
    M: torch.Tensor,
    F_batch: Callable[[torch.Tensor, float], torch.Tensor],
    buckets: list[torch.LongTensor],
    output_tensor: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Apply F column-wise to a CSC sparse matrix M, preserving sparsity.
    F takes a 1D tensor of length nnz_j and returns a tensor of the same shape,
    never introducing new nonzeros.

    If output_tensor is provided, it will be used to store the result inplace rather
    than allocating a new tensor.

    It can process input in batch by processing columns in groups (“buckets”) of
        similar sparsity so that each group can be batched.

        Parameters
        ----------
        M : torch.Tensor (CSC sparse)
            The input sparse matrix.
        F_batch : Callable
            A “batched” version of your 1D F: given a zero-padded [L×k] block of
            column-values and scalar z, returns its [L×k] projection.  Must
            preserve zero‐padding so no new nonzeros appear.
        buckets : list of 1D LongTensors
            Disjoint subsets of column indices, covering all columns of M.  Within
            each bucket, columns are zero-padded only up to that bucket’s max nnz.
        output_tensor : torch.Tensor, optional
            Pre-allocated values tensor to write into instead of allocating a new one.

        Returns
        -------
        torch.Tensor (CSC sparse)
            A sparse CSC matrix with the same sparsity pattern as M, where each
            column’s values have been replaced by F(col_values, z).
    """

    assert M.layout == torch.sparse_csc, "M must be a CSC sparse tensor"
    device, dtype = M.values().device, M.values().dtype
    ccol = M.ccol_indices()  # shape (n+1,)
    rowi = M.row_indices()  # shape (nnz,)
    vals = M.values()  # shape (nnz,)

    new_vals = torch.empty_like(vals)

    for cols in buckets:
        K = cols.numel()
        if K == 0:
            continue  # skip empty buckets

        # 1) compute starts/ends & lengths for this bucket. shape (K,)
        starts = ccol[cols].to(device)
        ends = ccol[cols + 1].to(device)
        lengths = ends - starts

        total = int(lengths.sum().item())

        # There should already be a check that no column is empty,
        # but extra check here that there are non-zero entries present
        if total == 0:
            continue

        # This is the highest number of non-zeroes of columns in the bucket
        L = int(lengths.max().item())

        # Compute cols_rep once, then derive all other indices via indexing
        # (avoids 2 extra repeat_interleave calls and a torch.cat)
        cols_rep = torch.arange(K, device=device).repeat_interleave(lengths)  # (total,)
        prefix = lengths.cumsum(0) - lengths  # shape (K,), avoids torch.cat
        idx_in_col = torch.arange(total, device=device) - prefix[cols_rep]
        flat_indices = starts[cols_rep] + idx_in_col

        # 2) build padded [L × K] block
        block = torch.zeros((L, K), device=device, dtype=dtype)
        block[idx_in_col, cols_rep] = vals[flat_indices]

        # 3) apply the batched projection
        proj_block = F_batch(block)  # returns shape (L, K)

        # 4) scatter back into new_vals
        new_vals[flat_indices] = proj_block[idx_in_col, cols_rep]

    # Rebuild and return
    if output_tensor is None:
        return torch.sparse_csc_tensor(ccol, rowi, new_vals, size=M.size())
    else:
        return output_tensor.values().copy_(new_vals)


def row_sums_csc(A: torch.Tensor) -> torch.Tensor:
    """
    Compute the sum over columns for each row of a CSC-format sparse matrix A,
    returning a dense 1-D tensor of length A.size(0).

    Args:
        A: torch.sparse_csc_tensor of shape (n_rows, n_cols)

    Returns:
        row_sums: 1-D dense tensor of length n_rows, where
                  row_sums[i] = sum_j A[i, j]
    """
    n_rows, _ = A.size()
    row_idx = A.row_indices()  # shape (nnz,)
    vals = A.values()  # shape (nnz,)

    # allocate output on same device and dtype
    row_sums = torch.zeros(n_rows, dtype=vals.dtype, device=vals.device)
    # scatter-add each nonzero into its row slot
    row_sums.scatter_add_(0, row_idx.to(torch.long), vals)
    return row_sums


def split_csc_by_cols(M: torch.Tensor, split_sizes: List[int]) -> List[torch.Tensor]:
    """
    Split a CSC-format sparse matrix M along its columns into blocks
    of widths given by split_sizes, preserving sparsity.

    Args:
        M:            torch.sparse_csc_tensor of shape (m, n)
        split_sizes:  list of positive ints summing to n

    Returns:
        List of torch.sparse_csc_tensor blocks, each of shape (m, split_sizes[i])
    """
    if M.layout != torch.sparse_csc:
        raise ValueError("M must be CSC-format sparse")

    m, n = M.size()
    if sum(split_sizes) != n:
        raise ValueError(f"split_sizes must sum to {n}")

    col_ptr = M.ccol_indices()  # shape (n+1,)
    row_idx = M.row_indices()  # shape (nnz,)
    vals = M.values()  # shape (nnz,)

    blocks = []
    col_offset = 0
    for width in split_sizes:
        # start/end pointers into col_ptr
        start_col = col_offset
        end_col = col_offset + width

        # nnz range for this block
        start_nnz = int(col_ptr[start_col].item())
        end_nnz = int(col_ptr[end_col].item())

        # slice out the per-block CSC arrays
        sub_col_ptr = (col_ptr[start_col : (end_col + 1)] - col_ptr[start_col]).clone()
        sub_row_idx = row_idx[start_nnz:end_nnz].clone()
        sub_vals = vals[start_nnz:end_nnz].clone()

        # build the sub‐matrix
        M_block = torch.sparse_csc_tensor(sub_col_ptr, sub_row_idx, sub_vals, size=(m, width))
        blocks.append(M_block)
        col_offset += width

    return blocks


def hstack_csc(tensors: list[torch.Tensor]) -> torch.Tensor:
    """
    Column‑wise concatenate a list of sparse CSC tensors.

    All tensors must
      • be `torch.sparse_csc_tensor`s,
      • share the same number of rows, dtype, and device.

    Returns
      A single CSC tensor whose columns are the columns of the inputs
      in the given order.
    """

    n_rows = tensors[0].size(0)
    dtype = tensors[0].dtype
    device = tensors[0].device

    for i, t in enumerate(tensors):
        if t.size(0) != n_rows:
            raise ValueError(f"tensor {i} has {t.size(0)} rows, expected {n_rows}")
        if t.dtype != dtype:
            raise TypeError("all tensors must share the same dtype")
        if t.device != device:
            raise TypeError("all tensors must be on the same device")

    ccol_chunks = []  # list of 1‑D tensors
    nnz_prefix = 0  # running count of non‑zeros
    total_cols = 0

    for t in tensors:
        ptr = t.ccol_indices()  # length = n_cols+1
        if nnz_prefix == 0:
            # keep the leading 0 for the very first block
            ccol_chunks.append(ptr)
        else:
            # drop the first 0, shift by current nnz offset
            ccol_chunks.append(ptr[1:] + nnz_prefix)

        nnz_prefix += t.values().shape[0]
        total_cols += t.size(1)

    new_ccol = torch.cat(ccol_chunks)

    # concatenate row‑indices and values
    new_row = torch.cat([t.row_indices() for t in tensors])
    new_vals = torch.cat([t.values() for t in tensors])

    result = torch.sparse_csc_tensor(
        new_ccol,
        new_row,
        new_vals,
        size=(n_rows, total_cols),
        dtype=dtype,
        device=device,
    )
    return result


def vstack_csc(tensors: list[torch.Tensor]) -> torch.Tensor:
    """
    Row-wise stack (vertical concatenate) a list of sparse CSC tensors.

    All tensors must
      • be `torch.sparse_csc_tensor`s,
      • share the same number of columns, dtype, and device.

    Returns
      A single CSC tensor whose rows are the rows of the inputs
      stacked vertically in the given order.
    """
    if not tensors:
        raise ValueError("Cannot stack empty list of tensors")

    n_cols = tensors[0].size(1)
    dtype = tensors[0].dtype
    device = tensors[0].device

    for i, t in enumerate(tensors):
        if t.layout != torch.sparse_csc:
            raise ValueError(f"tensor {i} must be CSC-format sparse")
        if t.size(1) != n_cols:
            raise ValueError(f"tensor {i} has {t.size(1)} columns, expected {n_cols}")
        if t.dtype != dtype:
            raise TypeError("all tensors must share the same dtype")
        if t.device != device:
            raise TypeError("all tensors must be on the same device")

    # For vertical stacking, we need to process column by column
    # and collect all nonzeros from all tensors for each column
    new_ccol = torch.zeros(n_cols + 1, dtype=torch.long, device=device)
    all_row_indices = []
    all_values = []

    # Process each column
    for col in range(n_cols):
        row_offset = 0

        # For each tensor, extract this column's nonzeros
        for t in tensors:
            ccol = t.ccol_indices()
            start_idx = int(ccol[col].item())
            end_idx = int(ccol[col + 1].item())

            if start_idx < end_idx:  # This column has nonzeros in this tensor
                col_rows = t.row_indices()[start_idx:end_idx] + row_offset
                col_vals = t.values()[start_idx:end_idx]

                all_row_indices.append(col_rows)
                all_values.append(col_vals)

            row_offset += t.size(0)

        # Update column pointer for next column
        new_ccol[col + 1] = sum(len(chunk) for chunk in all_row_indices)

    # Concatenate all collected data
    if all_row_indices:
        new_row = torch.cat(all_row_indices)
        new_vals = torch.cat(all_values)
    else:
        new_row = torch.tensor([], dtype=torch.long, device=device)
        new_vals = torch.tensor([], dtype=dtype, device=device)

    total_rows = sum(t.size(0) for t in tensors)

    result = torch.sparse_csc_tensor(
        new_ccol,
        new_row,
        new_vals,
        size=(total_rows, n_cols),
        dtype=dtype,
        device=device,
    )
    return result


def row_norms_csc(A: torch.Tensor) -> torch.Tensor:
    """
    Compute the L2-norm for each row of a CSC-format sparse matrix A,
    returning a dense 1-D tensor of length A.size(0).

    Args:
        A: torch.sparse_csc_tensor of shape (n_rows, n_cols)

    Returns:
        row_sums: 1-D dense tensor of length n_rows, where
                  row_sums[i] = sqrt(sum_j pow(A[i, j], 2))
    """
    n_rows, _ = A.size()
    row_idx = A.row_indices()  # shape (nnz,)
    vals = A.values().pow(2)  # shape (nnz,)

    # TODO: If this uses too much memory, we can look for a streaming approach
    # allocate output on same device and dtype
    row_sums = torch.zeros(n_rows, dtype=vals.dtype, device=vals.device)
    # scatter-add each nonzero into its row slot
    row_sums.scatter_add_(0, row_idx.to(torch.long), vals)
    return row_sums.pow(1 / 2)
