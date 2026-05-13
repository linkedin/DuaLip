import torch

from dualip.objectives.matching import MatchingInputArgs, MatchingSolverDualObjectiveFunction
from dualip.optimizers.agd import AcceleratedGradientDescent
from dualip.projections.base import create_projection_map

HOST_DEVICE = "cpu"


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
    ).to(HOST_DEVICE)

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
    ).to(HOST_DEVICE)

    b_vec = torch.tensor([0.7, 0.7, 0.7, 0.7, 0.7]).to(device=HOST_DEVICE)

    # The true complex constraint matrix ensures that sum_i a_compact[i,j]*c_compact[i,j] < b[j] for all j.
    # Using sparse CSC format, this can be represented by the entry-wise product and entry-wise inequality
    # a_compact.T * c_compact.T < b_vec in the matching problem schema.

    return a_compact.T.to_sparse_csc(), c_compact.T.to_sparse_csc(), b_vec


def test_simplex_solver_inequality():

    print("Running simplexInequality test")

    gamma = 1e-3

    a_expanded, c_expanded, b_vec = set_up_data_scala()
    J, num_items = a_expanded.shape

    # Use the new convenience function for constant projection types
    projection_map = create_projection_map("simplex", {"z": 1}, num_items)

    input_args = MatchingInputArgs(
        A=a_expanded,
        c=c_expanded,
        projection_map=projection_map,
        b_vec=b_vec,
        equality_mask=None,
    )
    objective = MatchingSolverDualObjectiveFunction(matching_input_args=input_args, gamma=gamma)

    initial_dual = 0.1 * torch.ones(5, device=HOST_DEVICE)

    solver = AcceleratedGradientDescent(max_iter=30)

    solver_result = solver.maximize(objective, initial_dual)

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


if __name__ == "__main__":

    test_simplex_solver_inequality()

    print("All tests completed succesfully")
