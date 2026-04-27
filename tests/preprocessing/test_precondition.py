import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from dualip.preprocessing.precondition import (
    dist_jacobi_precondition,
    jacobi_invert_precondition,
    jacobi_precondition,
)
from dualip.utils.sparse_utils import split_csc_by_cols

ccol_indices = [0, 2, 3, 5, 8, 10, 12, 15, 16]
row_indices = [2, 3, 3, 1, 2, 0, 1, 2, 0, 2, 0, 3, 1, 2, 3, 2]
values = [
    0.2617,
    0.3848,
    0.2617,
    0.8047,
    0.4121,
    0.7383,
    0.3555,
    0.3418,
    0.5469,
    0.9570,
    0.3555,
    0.6523,
    0.1738,
    0.4121,
    0.9375,
    0.3008,
]


A_test = torch.sparse_csc_tensor(
    torch.tensor(ccol_indices, dtype=torch.int64),
    torch.tensor(row_indices, dtype=torch.int32),
    torch.tensor(values),
    dtype=torch.float32,
)

b_test = torch.tensor([1, 2, 3, 4], dtype=torch.float32)


@pytest.fixture(scope="module")
def norms_path(tmp_path_factory):
    """Temporary file for persisting the norm vector."""
    path = tmp_path_factory.mktemp("norms") / "row_norms.pt"
    return str(path)


def test_precondition_saves_norms(norms_path):
    """Row norms are computed and persisted correctly."""
    jacobi_precondition(A_test.clone(), b_test.clone(), norms_save_path=norms_path)

    norms_path = Path(norms_path)
    assert norms_path.exists(), "Norm file was not created"
    saved = torch.load(norms_path)
    expected = A_test.to_dense().norm(2, 1)
    assert torch.allclose(saved, expected), "Saved norms differ from reference"


def test_precondition_scaling(norms_path):
    """A and b are scaled by 1 / row_norms."""

    A_scaled = A_test.clone()
    b_scaled = b_test.clone()

    jacobi_precondition(A_scaled, b_scaled, norms_save_path=norms_path)

    row_norms = torch.load(norms_path)
    reciprocal = 1.0 / row_norms

    # Dense check to avoid implementing sparse comparison utility here
    expected_A_dense = reciprocal.unsqueeze(1) * A_test.to_dense()
    assert torch.allclose(A_scaled.to_dense(), expected_A_dense)

    expected_b = reciprocal * b_test
    assert torch.allclose(b_scaled, expected_b)


def test_invert_precondition(norms_path):
    """
    Jacobi scaling effectively scales the dual variable by diag(row_norms).
    Hence mapping the ones vector to the original space should be 1/row_norms
    """

    A_scaled = A_test.clone()
    b_scaled = b_test.clone()

    dual_val = torch.tensor([1, 1, 1, 1], dtype=torch.float32)

    jacobi_precondition(A_scaled, b_scaled, norms_save_path=norms_path)

    restored = jacobi_invert_precondition(dual_val, norms_path)

    row_norms = A_test.to_dense().norm(2, 1)
    reciprocal = 1.0 / row_norms

    assert torch.allclose(restored, reciprocal)


# ---------------------------------------------------------------------------
# Distributed Jacobi preconditioner tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif("RANK" in os.environ, reason="Only runs in non-distributed (no torchrun) context")
def test_dist_jacobi_precondition_raises_without_dist():
    """dist_jacobi_precondition must raise when torch.distributed is not initialized."""
    assert not dist.is_initialized(), "Expected dist to be uninitialized in this test"
    with pytest.raises(RuntimeError, match="torch.distributed"):
        dist_jacobi_precondition(A_test.clone(), b_test.clone())


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 CUDA GPUs")
@pytest.mark.skipif(
    "RANK" not in os.environ,
    reason="Requires torchrun - run with: torchrun --nproc_per_node=2 -m pytest ...",
)
def test_dist_jacobi_precondition():
    """
    Distributed preconditioning gives the same row norms, A scaling, and b
    scaling as the single-process version when data is split across 2 ranks.
    """
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = f"cuda:{rank}"

        A_full = A_test.to(device)
        b_full = b_test.to(device)

        expected_row_norms = A_full.to_dense().norm(2, dim=1)

        n_cols = A_full.size(1)
        split_sizes = [
            n_cols // world_size + (1 if i < n_cols % world_size else 0)
            for i in range(world_size)
        ]
        a_splits = split_csc_by_cols(A_full, split_sizes)

        A_local = a_splits[rank].to(device)
        original_vals = A_local.values().clone()
        original_row_idx = A_local.row_indices().clone()

        b_local = b_full.clone() if rank == 0 else None

        row_norms = dist_jacobi_precondition(A_local, b_local)

        assert torch.allclose(row_norms, expected_row_norms, atol=1e-5), (
            f"Rank {rank}: row norms mismatch"
        )

        expected_vals = original_vals / expected_row_norms[original_row_idx.to(torch.long)]
        assert torch.allclose(A_local.values(), expected_vals, atol=1e-5), (
            f"Rank {rank}: A_local values not scaled correctly"
        )

        if rank == 0:
            expected_b = b_full / expected_row_norms
            assert torch.allclose(b_local, expected_b, atol=1e-5), (
                "Rank 0: b not scaled correctly"
            )

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 CUDA GPUs")
@pytest.mark.skipif(
    "RANK" not in os.environ,
    reason="Requires torchrun - run with: torchrun --nproc_per_node=2 -m pytest ...",
)
def test_dist_jacobi_precondition_saves_norms(tmp_path):
    """Rank 0 saves row norms to disk; the saved norms match the global row norms."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = f"cuda:{rank}"

        norms_path = str(tmp_path / "row_norms.pt") if rank == 0 else None

        A_full = A_test.to(device)
        n_cols = A_full.size(1)
        split_sizes = [
            n_cols // world_size + (1 if i < n_cols % world_size else 0)
            for i in range(world_size)
        ]
        a_splits = split_csc_by_cols(A_full, split_sizes)
        A_local = a_splits[rank].to(device)
        b_local = b_test.to(device).clone() if rank == 0 else None

        row_norms = dist_jacobi_precondition(A_local, b_local, norms_save_path=norms_path)

        if rank == 0:
            assert Path(norms_path).exists(), "Rank 0 did not save the norm file"
            saved = torch.load(norms_path, map_location=device)
            assert torch.allclose(saved, row_norms, atol=1e-5), (
                "Saved norms differ from returned row norms"
            )

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
