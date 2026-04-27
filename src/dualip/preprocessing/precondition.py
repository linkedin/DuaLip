from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist

from dualip.utils.sparse_utils import left_multiply_sparse, row_norms_csc


def jacobi_precondition(A: torch.sparse_csc_tensor, b: torch.Tensor, norms_save_path: str = None):
    """
    Scale each row of A (and b) in place by the reciprocal of the row L2-norms

    If ``norms_save_path`` is given, the row-norm vector is saved so the
    scaling can be undone later with ``jacobi_invert_precondition``.
    Returns the same (rescaled) A and the scaled b in place.
    """
    row_norms = row_norms_csc(A)

    if norms_save_path:

        path = Path(norms_save_path)
        torch.save(row_norms, path)

    reciprocal = 1 / row_norms

    left_multiply_sparse(reciprocal, A, A)

    b.mul_(reciprocal)
    return row_norms


def dist_jacobi_precondition(
    A_local: torch.Tensor,
    b: Optional[torch.Tensor],
    norms_save_path: Optional[str] = None,
) -> torch.Tensor:
    """
    Distributed Jacobi preconditioning for a column-partitioned matrix.

    Each rank holds a column slice ``A_local`` of shape ``(m, n_local)``.
    The full row L2-norms are reconstructed via an ``all_reduce(SUM)`` over
    the per-rank partial sums of squared values, then each rank scales its
    local slice in-place. Rank 0 also scales ``b`` when provided.

    Requires ``torch.distributed`` to be initialized before calling.

    Parameters
    ----------
    A_local : torch.sparse_csc_tensor
        Local column slice of the constraint matrix, shape ``(m, n_local)``.
        Modified in-place.
    b : torch.Tensor or None
        Right-hand-side vector of length ``m``. Pass the actual tensor on
        rank 0 and ``None`` on all other ranks. Modified in-place when given.
    norms_save_path : str, optional
        If provided, rank 0 saves the row-norm tensor to this path so it can
        be used later with :func:`jacobi_invert_precondition`.

    Returns
    -------
    torch.Tensor
        Dense 1-D tensor of length ``m`` with the full (global) row L2-norms.
    """
    if not dist.is_initialized():
        raise RuntimeError(
            "dist_jacobi_precondition requires torch.distributed to be initialized. "
            "Call torch.distributed.init_process_group() before using this function."
        )

    n_rows = A_local.size(0)
    row_idx = A_local.row_indices()
    vals = A_local.values()

    local_sq = torch.zeros(n_rows, dtype=vals.dtype, device=vals.device)
    local_sq.scatter_add_(0, row_idx.to(torch.long), vals.pow(2))

    dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)
    row_norms = local_sq.pow(0.5)

    if dist.get_rank() == 0 and norms_save_path:
        torch.save(row_norms, Path(norms_save_path))

    reciprocal = 1.0 / row_norms
    left_multiply_sparse(reciprocal, A_local, A_local)

    if b is not None:
        b.mul_(reciprocal)

    return row_norms


def jacobi_invert_precondition(dual_val: torch.Tensor, norms_path_or_tensor: str | torch.Tensor):
    """
    Reverse the Jacobi pre-conditioning using row-norms saved on disk.

    Given a dual_val (lambda) in the pre-conditioned space, this function multiplies by
    diag(1/row_norms) to map it back to the original scaling. This is because scaling
    (Ax - b) by diagonal matrix D effectively scales lambda by D^{-1}.

    Parameters
    ----------
    dual_val : torch.Tensor
        Dual variable value in preconditioned LP
    norms_path_or_tensor : str or torch Tensor
        Either path where :func:`jacobi_precondition` persisted the row-norm tensor or
        the row-norm tensor itself.

    Returns
    -------
    torch.Tensor
        The dual vector in the original scaling.
    """

    if isinstance(norms_path_or_tensor, str):
        path = Path(norms_path_or_tensor)
        row_norms = torch.load(path, map_location=dual_val.device)

    if isinstance(norms_path_or_tensor, torch.Tensor):
        row_norms = norms_path_or_tensor.to(dual_val.device)

    return (1 / row_norms) * dual_val
