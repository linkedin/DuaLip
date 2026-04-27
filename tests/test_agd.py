import torch

from dualip.objectives.base import ObjectiveResult
from dualip.optimizers.agd import AcceleratedGradientDescent

HOST_DEVICE = "cpu"


class Quadratic1DObjective:
    def __init__(self):
        self.dual_dimensionality = 1
        self.equality_mask: bool = None

    def calculate(self, dual_val: torch.tensor, save_primal=False, **kwargs):
        """
        Implements the 1D quadratic objective:
          f(x) = -(x - 3.0)^2
        Its gradient is:
          grad = -2.0 * (x - 3.0)
        For x = 0, the gradient is 6.0.
        """
        x = dual_val[0]
        obj = -((x - 3.0) ** 2)
        grad = torch.tensor([-2.0 * (x - 3.0)]).to(dual_val.device)
        return ObjectiveResult(dual_gradient=grad, dual_objective=obj, reg_penalty=None)


class SimpleObjective:
    def __init__(self):
        # This dummy objective is 2-dimensional.
        self.dual_dimensionality = 2
        self.equality_mask: bool = None

    def calculate(self, dual_val: torch.tensor, save_primal=False, **kwargs):
        """
        Implements the 2D objective:
          f(x, y) = -(x - 3)^2 - (y + 5)^2
        with gradient:
          [ -2*(x-3), -2*(y+5) ]
        For dual = [0, 0] the gradient is [6, -10].
        """
        x, y = dual_val
        obj = -((x - 3.0) ** 2) - (y + 5.0) ** 2
        grad = torch.tensor([-2.0 * (x - 3.0), -2.0 * (y + 5.0)]).to(dual_val.device)
        return ObjectiveResult(dual_gradient=grad, dual_objective=obj, reg_penalty=None)


def test_quadratic_1d_function():
    # For Quadratic1DObjective, the initial gradient is 6.0.
    # So after one iteration, the solution should be at 6.0 * initial_step_size.
    initial_gradient = 6.0
    default_step_size = 1e-5

    # Test with the default initial_step_size.
    solver_default = AcceleratedGradientDescent(max_iter=1)
    solver_default_result = solver_default.maximize(Quadratic1DObjective(), torch.tensor([0.0], device=HOST_DEVICE))
    assert abs(solver_default_result.dual_val[0] - (initial_gradient * default_step_size)) < 1e-10, (
        f"Test fails for default initialStepSize: expected {initial_gradient * default_step_size}, "
        f"got {solver_default_result.dual_val[0]}"
    )

    # Test with a new initial_step_size.
    new_step_size = 0.1
    solver_new_step_size = AcceleratedGradientDescent(max_iter=1, initial_step_size=new_step_size)
    solver_new_step_size_result = solver_new_step_size.maximize(
        Quadratic1DObjective(), torch.tensor([0.0], device=HOST_DEVICE)
    )
    assert abs(solver_new_step_size_result.dual_val[0] - (initial_gradient * new_step_size)) < 1e-10, (
        f"Test fails for new initialStepSize: expected {initial_gradient * new_step_size}, "
        f"got {solver_new_step_size_result.dual_val[0]}"
    )

    print(
        "Quadratic1DObjective test passed. Final solutions:",
        solver_default_result,
        solver_new_step_size_result,
    )


def test_simple_objective_dual_value():
    # For SimpleObjective starting at [0, 0]:
    # The gradient is [6, -10] from the objective (since -2*(0-3)=6, -2*(0+5)=-10).
    # Our stub calculate_step_size returns a constant step size of 1e-5.
    # The dual objective f(x,y) = -(x-3)^2 - (y+5)^2 is initially f(0,0) = -9 - 25 = -34.
    # With a very small step, the dual objective value will increase slightly toward the optimum f(3,0) = -25.
    default_step_size = 1e-5

    solver = AcceleratedGradientDescent(max_iter=30, initial_step_size=default_step_size)
    solver_result = solver.maximize(SimpleObjective(), torch.tensor([0.0, 0.0], device=HOST_DEVICE))
    for i, (dual, step) in enumerate(zip(solver_result.dual_objective_log[:25], solver_result.step_size_log[:25])):
        print(f"Iteration: {i + 1}.  Dual: {dual}.   Step: {step}")

    # Dual objective values at selected iterations to check the entire optimization trace
    # matches the scala version up to high precision.
    true_values = [
        (2, -33.9996400036),
        (16, -28.60551547593112),
        (23, -25.473701313626133),
        (29, -25.00382134903756),
    ]

    for i, true_val in true_values:
        # QQ Note: increase tol to 1e-7 for torch
        assert abs(solver_result.dual_objective_log[i - 1] - true_val) < 1e-5, (
            f"SimpleObjective has incorrect value at iteration {i+1}"
            f" expected dual objective value {true_val} but computed {solver_result.dual_objective_log[i-1]}"
        )

    print("SimpleObjective dual value test passed.")
