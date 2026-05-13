import os

import pytest
import torch

from dualip.objectives.matching import MatchingInputArgs, MatchingSolverDualObjectiveFunctionDistributed
from dualip.optimizers.agd import AcceleratedGradientDescent
from dualip.projections.base import create_projection_map
from dualip.utils.dist_utils import global_to_local_projection_map, split_tensors_to_devices


@pytest.fixture(scope="module")
def init_distributed():
    """Initialize torch.distributed if running under torchrun."""
    import torch.distributed as dist

    # Check if torchrun set the environment variables
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        yield
        # Don't destroy - torchrun manages the lifecycle
    else:
        # Not running under torchrun, skip distributed tests
        yield


def set_up_data_scala():
    # a(i, j): cost associated with user i for item j
    a_compact = torch.tensor(
        [
            [
                0.307766110869125,
                0.483770735096186,
                0.624996477039531,
                0.669021712383255,
                0.535811153938994,
            ],
            [
                0.257672501029447,
                0.812402617651969,
                0.882165518123657,
                0.204612161964178,
                0.710803845431656,
            ],
            [
                0.552322433330119,
                0.370320537127554,
                0.28035383997485,
                0.357524853432551,
                0.538348698290065,
            ],
            [
                0.0563831503968686,
                0.546558595029637,
                0.398487901547924,
                0.359475114848465,
                0.74897222686559,
            ],
            [
                0.468549283919856,
                0.170262051047757,
                0.76255108229816,
                0.690290528349578,
                0.420101450523362,
            ],
        ]
    ).to("cpu")

    # c(i, j): objective function coefficient associated with user i for item j
    c_compact = torch.tensor(
        [
            [
                -0.307766110869125,
                -0.483770735096186,
                -0.624996477039531,
                -0.669021712383255,
                -0.535811153938994,
            ],
            [
                -0.257672501029447,
                -0.812402617651969,
                -0.882165518123657,
                -0.204612161964178,
                -0.710803845431656,
            ],
            [
                -0.552322433330119,
                -0.370320537127554,
                -0.28035383997485,
                -0.357524853432551,
                -0.538348698290065,
            ],
            [
                -0.0563831503968686,
                -0.546558595029637,
                -0.398487901547924,
                -0.359475114848465,
                -0.74897222686559,
            ],
            [
                -0.468549283919856,
                -0.170262051047757,
                -0.76255108229816,
                -0.690290528349578,
                -0.420101450523362,
            ],
        ]
    ).to("cpu")

    b_vec = torch.tensor([0.7, 0.7, 0.7, 0.7, 0.7]).to(device="cpu")

    return a_compact.T.to_sparse_csc(), c_compact.T.to_sparse_csc(), b_vec


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    "RANK" not in os.environ, reason="Requires torchrun - run with: torchrun --nproc_per_node=2 -m pytest ..."
)
def test_simplex_solver_inequality_distributed(init_distributed):
    """
    Test distributed matching objective with multi-GPU setup.

    This test requires torch.distributed to be initialized.
    Run with:
        torchrun --nproc_per_node=2 -m pytest \
            tests/distributed/test_matching_distributed.py::test_simplex_solver_inequality_distributed -v

    When run without torchrun, this test will be skipped.
    """
    import torch.distributed as dist

    print("Running simplexInequality distributed test")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    gamma = 1e-3
    a_expanded, c_expanded, b_vec = set_up_data_scala()
    J, num_items = a_expanded.shape

    host_device = "cuda:0"  # Aggregation device
    compute_devices = [f"cuda:{i}" for i in range(world_size)]

    # Use the new convenience function for constant projection types
    projection_map = create_projection_map("simplex", {"z": 1}, num_items)

    # Split data across GPUs
    A_splits, c_splits, split_index_map = split_tensors_to_devices(a_expanded, c_expanded, compute_devices)

    # Each rank takes its partition
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    A_local = A_splits[rank].to(device)
    c_local = c_splits[rank].to(device)
    pm_local = global_to_local_projection_map(projection_map, split_index_map[rank])

    # Create local input args (b_vec=None for local partition)
    local_input_args = MatchingInputArgs(
        A=A_local,
        c=c_local,
        projection_map=pm_local,
        b_vec=None,
        equality_mask=None,
    )

    # Create distributed objective with new API
    f = MatchingSolverDualObjectiveFunctionDistributed(
        local_matching_input_args=local_input_args,
        b_vec=b_vec,
        gamma=gamma,
        host_device=host_device,
    )

    initial_dual = 0.1 * torch.ones(5, device=device)

    solver = AcceleratedGradientDescent(max_iter=30)
    solver_result = solver.maximize(f, initial_dual, rank=rank)

    # Only rank 0 checks results
    if rank == 0:
        true_values = [
            (2, -3.6010155991401818),
            (16, -3.60842718733725),
            (23, -3.5080258013053136),
            (29, -3.4868496294227143),
        ]

        for i, true_val in true_values:
            assert abs(solver_result.dual_objective_log[i - 1] - true_val) < 1e-5, (
                f"Solution has incorrect dual objective at iteration {i+1}"
                f" expected dual objective value {true_val} but computed {solver_result.dual_objective_log[i-1]}"
            )
        print("  ✓ All assertions passed on rank 0")
