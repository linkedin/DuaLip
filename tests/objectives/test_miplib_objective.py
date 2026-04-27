import torch

from dualip.objectives.miplib import MIPLIB2017ObjectiveFunction, MIPLIBInputArgs
from dualip.optimizers.agd import AcceleratedGradientDescent
from dualip.projections import ProjectionEntry, create_projection_map


def test_miplib_general_convergence_criteria():
    """
    Test for the PDLP convergence criteria for general bounds.
    """

    A = torch.tensor(
        [
            [1.0, 1.0, 1.0, 0.0],  # x1 + x2 + x3 <= 5
            [2.0, -1.0, 0.0, 1.0],  # 2x1 - x2 + x4 <= 3
            [-1.0, 0.0, 4.0, -1.0],  # -x1 + 4x3 - x4 <= 2
        ]
    )

    # Optimal x: [ 0.  1. -0. -2.]

    b = torch.tensor(([5.0, 3.0, 2.0]))
    c = torch.tensor([2.0, 3.0, -1.0, 4.0])

    equality_mask = torch.tensor([False, False, False])

    projection_map = {
        "bound_1": ProjectionEntry("box", {"l": 0.0, "u": 3.0}, indices=[0]),
        "bound_2": ProjectionEntry("box", {"l": 1.0, "u": 4.0}, indices=[1]),
        "bound_3": ProjectionEntry("box", {"l": 0.0, "u": float("nan")}, indices=[2]),
        "bound_4": ProjectionEntry("box", {"l": -2.0, "u": 2.0}, indices=[3]),
    }

    input_args = MIPLIBInputArgs(
        A=A,
        c=c,
        projection_map=projection_map,
        b_vec=b,
        equality_mask=equality_mask,
    )
    objective = MIPLIB2017ObjectiveFunction(miplib_input_args=input_args)

    # Should pass the primal-dual gap check due to optimal dual value
    optimal_dual = torch.tensor([+0.0000, +0.0000, +0.2500])
    _, _, _, _, converged = objective.calculate_convergence_bound(optimal_dual, tol=1e-5)
    assert converged

    # Should pass the check due to high tol
    dual_val = torch.tensor([0.0, -0.01, 0.26])
    _, _, _, _, converged = objective.calculate_convergence_bound(dual_val, tol=1e-1)
    assert converged

    # Should fail the check due to low tol
    dual_val = torch.tensor([0.0, -0.01, 0.26])
    _, _, _, _, converged = objective.calculate_convergence_bound(dual_val, tol=1e-5)
    assert not converged


def test_miplib_general_convergence_criteria_II():
    """
    Same test as the unit box convergence criteria.
    """

    c = torch.tensor([1.0, 1.0])
    A = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    b = torch.tensor([1.0, 3.0])
    equality_mask = torch.tensor([False, False])

    projection_map = create_projection_map("box", {"l": 0.0, "u": 1.0}, 2, indices=[0, 1])

    input_args = MIPLIBInputArgs(
        A=A,
        c=c,
        projection_map=projection_map,
        b_vec=b,
        equality_mask=equality_mask,
    )
    objective = MIPLIB2017ObjectiveFunction(miplib_input_args=input_args)

    # Should fail the primal-dual gap check
    dual_val = 0.1 * torch.ones(2)
    _, _, _, _, converged = objective.calculate_convergence_bound(dual_val, tol=1e-5)
    assert not converged

    # Should pass the check due to high tol
    dual_val = 0.1 * torch.ones(2)
    _, _, _, _, converged = objective.calculate_convergence_bound(dual_val, tol=1)
    assert converged

    # Should pass the check due to optimal dual val
    dual_val = torch.tensor([0.0, 0.0])
    _, _, _, _, converged = objective.calculate_convergence_bound(dual_val, tol=1e-8)
    assert converged


def test_miplib_general_convergence_criteria_III():
    """
    Same test as the unit box convergence criteria, but with negative objective.
    """

    c = torch.tensor([-1.0, -1.0])
    A = torch.tensor([[4.0, 1.0], [1.0, 2.0]])
    b = torch.tensor([2.0, 1.0])
    equality_mask = None

    projection_map = create_projection_map("box", {"lower": 0.0, "upper": 1.0}, 2, indices=[0, 1])
    input_args = MIPLIBInputArgs(
        A=A,
        c=c,
        projection_map=projection_map,
        b_vec=b,
        equality_mask=equality_mask,
    )
    objective = MIPLIB2017ObjectiveFunction(miplib_input_args=input_args, gamma=0.001)
    solver = AcceleratedGradientDescent(
        max_iter=500,
        save_primal=True,
    )
    initial_dual = torch.zeros(2)
    solver_result = solver.maximize(objective, initial_dual)

    # Should pass the primal-dual gap check
    dual_val = torch.tensor([0.14285714, 0.42857143])
    x = solver_result.objective_result.primal_var
    _, _, _, _, converged = objective.calculate_convergence_bound(dual_val, x=x, tol=1e-4)
    assert converged


def test_miplib_convergence_with_one_sided_x_bound_I():
    """
    Same test as the unit box convergence criteria, but with negative objective.
    """

    c = torch.tensor([-1.0, -1.0])
    A = torch.tensor([[4.0, 1.0], [1.0, 2.0]])
    b = torch.tensor([2.0, 1.0])
    equality_mask = None

    projection_map = create_projection_map("cone", {"lower": 0.0}, 2, indices=[0, 1])
    input_args = MIPLIBInputArgs(
        A=A,
        c=c,
        projection_map=projection_map,
        b_vec=b,
        equality_mask=equality_mask,
    )
    objective = MIPLIB2017ObjectiveFunction(miplib_input_args=input_args, gamma=0.001)
    solver = AcceleratedGradientDescent(
        initial_step_size=1e-6,
        max_step_size=1e-5,
        max_iter=10000,
        save_primal=True,
    )
    initial_dual = torch.zeros(2)
    solver_result = solver.maximize(objective, initial_dual)
    x = solver_result.objective_result.primal_var

    # Should pass the primal-dual gap check
    dual_val = torch.tensor([0.14285714, 0.42857143])
    _, _, _, _, converged = objective.calculate_convergence_bound(dual_val, x=x, tol=1e-3)
    assert converged  # lower tolerance requires increaseing max_iter above


def test_miplib_convergence_with_one_sided_x_bound_II():
    """
    Same test as the unit box convergence criteria, but with negative objective.
    """

    c = torch.tensor([-1.0, -1.0])
    A = torch.tensor([[4.0, 1.0], [1.0, 2.0]])
    b = torch.tensor([2.0, 1.0])
    equality_mask = None

    projection_map = create_projection_map("cone", {"upper": 1.0}, 2, indices=[0, 1])
    input_args = MIPLIBInputArgs(
        A=A,
        c=c,
        projection_map=projection_map,
        b_vec=b,
        equality_mask=equality_mask,
    )
    objective = MIPLIB2017ObjectiveFunction(miplib_input_args=input_args, gamma=0.001)
    solver = AcceleratedGradientDescent(
        initial_step_size=1e-6,
        max_step_size=1e-5,
        max_iter=10000,
        save_primal=True,
    )
    initial_dual = torch.zeros(2)
    solver_result = solver.maximize(objective, initial_dual)
    x = solver_result.objective_result.primal_var

    # Should pass the primal-dual gap check
    dual_val = torch.tensor([0.14285714, 0.42857143])
    _, _, _, _, converged = objective.calculate_convergence_bound(dual_val, x=x, tol=1e-3)
    assert converged
