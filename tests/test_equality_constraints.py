import torch

from dualip.objectives.miplib import MIPLIB2017ObjectiveFunction, MIPLIBInputArgs
from dualip.optimizers.agd import AcceleratedGradientDescent, project_on_nn_cone
from dualip.projections.base import create_projection_map


def test_project_on_nn_cone():
    y = torch.tensor([-1.0, -1.0, 2.0, -3.0, 4.0])
    equality_mask = torch.tensor([False, True, False, True, False])

    projection = project_on_nn_cone(y, equality_mask)
    assert (
        projection == torch.tensor([0.0, -1.0, 2.0, -3.0, 4.0])
    ).all(), "Projection failed for nonnegative cone with equality constraints"


def test_solver_with_equality_constraint():
    """
    Test solving a simple LP:

        minimize   x1 + 2 * x2
        subject to x1 + x2 = 4
                   0 <= x1 <= 1
                   0 <= x2

    Optimal solution:
        x1 = 1, x2 = 3 → optimal value = 7.0
    """
    device = torch.device("cpu")

    A = torch.tensor([[1.0, 1.0]], device=device)  # equality constraint: x1 + x2 = 4
    c = torch.tensor([1.0, 2.0], device=device)  # objective: x1 + 2*x2
    b_vec = torch.tensor([4.0], device=device)

    gamma = 1e-5
    equality_mask = torch.tensor([True], device=device)
    projection_map = create_projection_map("box", {"upper": 1}, num_indices=2, indices=[0])
    initial_dual = torch.tensor([0.0], device=device)

    input_args = MIPLIBInputArgs(
        A=A,
        c=c,
        projection_map=projection_map,
        b_vec=b_vec,
        equality_mask=equality_mask,
    )
    objective = MIPLIB2017ObjectiveFunction(
        miplib_input_args=input_args,
        gamma=gamma,
    )

    solver = AcceleratedGradientDescent(max_iter=1000)
    solver_result = solver.maximize(objective, initial_dual)

    # Verify the solution is correct within tolerance
    expected_value = torch.tensor(7.0, device=device)
    actual_value = torch.tensor(solver_result.dual_objective, device=device)

    assert torch.isclose(
        actual_value, expected_value, atol=1e-5
    ), f"Expected objective ~7.0, but got {actual_value.item():.6f}"
