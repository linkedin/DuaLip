# DuaLip: Dual Decomposition-based Linear Program Solver

[![License](https://img.shields.io/badge/License-BSD_2--Clause-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6+](https://img.shields.io/badge/PyTorch-2.6%2B-ee4c2c.svg)](https://pytorch.org/)
[![Docs](https://img.shields.io/badge/docs-GitHub_Pages-green.svg)](https://linkedin.github.io/DuaLip)

DuaLip is an **extreme-scale Linear Program (LP) solver** built on PyTorch. It solves structured LP problems of the form:

```
minimize    c^T x
subject to  Ax ≤ b
            x_i ∈ C_i   for all i ∈ {1, 2, ..., I}
```

where `x = (x_1, ..., x_I)` is the full vector of optimization variables, `x_i` is the vector associated with entity `i` (e.g., a user), and `A, b, c, C_i` are problem-specific data.

Many large-scale decision systems — allocation, assignment, marketplace shaping — reduce to solving very large LPs on a recurring cadence. General-purpose LP solvers treat the constraint matrix as an unstructured operator and have not scaled to these extreme problem sizes. Specialized matching solvers cannot accommodate the heterogeneous constraint families (budgets, pacing, fairness, frequency caps) that arise in production.

DuaLip fills this gap with a **ridge-regularized dual ascent** approach that:
- Exploits block-diagonal structure for massive parallelism
- Requires no commercial solver license
- Runs entirely on commodity GPUs with open-source PyTorch

## Features

- **Extreme scale** — structured LPs with hundreds of millions of entities and up to trillions of variables
- **GPU-accelerated** — significant wall-clock speedups over the prior Spark-based solver through native CUDA execution
- **Operator-centric API** — new LP formulations require only a new `ObjectiveFunction`; projections and the solve loop are reused
- **Distributed multi-GPU** — scales horizontally via `torch.distributed` (NCCL); communication cost depends only on the dual dimension
- **Nesterov-accelerated gradient ascent** — with Lipschitz-based step sizing, Jacobi preconditioning, and optional γ decay
- **Polytope projections** — box, simplex, and cone constraints, extensible via a registry pattern
- **Warm start** — initialize from a previous dual solution for faster convergence on recurring workloads
- **MLflow integration** — track dual objective, step sizes, hyperparameters, and convergence diagnostics
- **Python-native** — aligns with common numerical and ML stacks for easy instrumentation and debugging

## Quick start

### Requirements

- Python >= 3.10
- PyTorch >= 2.6.0

### Installation

```bash
pip install -e .
```

### Usage

```python
import torch
from dualip.objectives.matching import MatchingInputArgs
from dualip.projections.base import create_projection_map
from dualip.run_solver import run_solver
from dualip.types import ComputeArgs, ObjectiveArgs, SolverArgs

# Build your problem data (A and c in CSC format, budget vector b)
# and a projection map defining per-column constraints
input_args = MatchingInputArgs(
    A=A_csc,                            # sparse constraint matrix (CSC)
    c=C_csc,                            # sparse cost matrix (CSC)
    b_vec=b_vec,                        # budget / RHS vector
    projection_map=create_projection_map(projection_entries),
)

solver_args = SolverArgs(max_iter=5000, gamma=1e-3)
compute_args = ComputeArgs(host_device="cuda:0")
objective_args = ObjectiveArgs(objective_type="matching")

result = run_solver(input_args, solver_args, compute_args, objective_args)

print(f"Dual objective: {result.dual_objective:.6f}")
```

For a complete end-to-end example, see [examples/movielens_matching/movies_lens_matching.py](examples/movielens_matching/movies_lens_matching.py) which solves a matching problem on the MovieLens dataset. An example using a MIPLIB 2017 problem is available under [examples/miplib_2017/](examples/miplib_2017/).

For reference on setting up your own problem, see [benchmark/run_matching_benchmark.py](benchmark/run_matching_benchmark.py) (single-GPU) and [benchmark/run_matching_benchmark_dist.py](benchmark/run_matching_benchmark_dist.py) (distributed multi-GPU).

## PyTorch-based solver

DuaLip was originally built on Scala/Spark. The PyTorch rewrite is a ground-up redesign that replaces the CPU-centric, schema-bound runtime with a GPU-native, composable architecture, delivering significant wall-clock speedups while making new problem formulations far easier to express.

The solver uses an **operator-centric programming model** built around three decoupled primitives — `ObjectiveFunction`, `ProjectionMap`, and `Maximizer` — so new LP formulations only require implementing a new `ObjectiveFunction` while reusing the solve loop, projections, and distributed primitives.

**Key design choices for GPU performance:**
- Sparse CSC storage aligned with block-diagonal structure
- Batched projection kernels
- Jacobi preconditioning with γ continuation schedule
- Distributed multi-GPU via `torch.distributed` (NCCL)
- Warm start support
- Pure PyTorch — no custom C++/CUDA kernels

## Documentation

Full documentation is available at **[linkedin.github.io/DuaLip](https://linkedin.github.io/DuaLip)**.

## Migrating from the Spark-based solver

This PyTorch-based solver is a ground-up rewrite of the original [Spark/Scala DuaLip solver](https://arxiv.org/abs/2103.05277). The core dual-decomposition algorithm is the same, but the implementation, data formats, and APIs are entirely new.

**Version history:** Tags `v1.0.0` through `v4.0.5` correspond to the Spark/Scala solver. The PyTorch rewrite begins at `v5.0.1` on the `master` branch.

See the [Quick start](#quick-start) section and [examples/movielens_matching/movies_lens_matching.py](examples/movielens_matching/movies_lens_matching.py) for examples of the new API.

## Development

1. **Create and activate a virtual environment**

   ```bash
   python -m venv .venv && source .venv/bin/activate
   ```

2. **Install dev dependencies and pre-commit hooks**

   ```bash
   make install
   ```

3. **Run tests**

   ```bash
   make test
   ```

4. **Run checkstyle** (format + lint)

   ```bash
   make checkstyle
   ```

## Repository structure

```
src/dualip/          Core library
  objectives/        Objective functions (matching, MIPLIB)
  optimizers/        Accelerated gradient descent solver
  projections/       Box, simplex, cone projections
  preprocessing/     Input validation and Jacobi preconditioning
  utils/             Sparse, distributed, and MLflow utilities
tests/               Unit and distributed tests
examples/            Example problems (MovieLens matching, MIPLIB 2017)
benchmark/           Scaling and performance benchmarks
docs/                Sphinx documentation source
```

## Citing DuaLip

If you use DuaLip in your work, please cite:

```bibtex
@inproceedings{ramanath:21,
  author  = {Ramanath, Rohan and Keerthi, Sathiya S. and Basu, Kinjal and Salomatin, Konstantin and Yao, Pan},
  title   = {Efficient Algorithms for Global Inference in Internet Marketplaces},
  journal = {arXiv preprint arXiv:2103.05277},
  year    = {2021},
  url     = {https://arxiv.org/abs/2103.05277}
}

@InProceedings{pmlr-v119-basu20a,
  title     = {{ECLIPSE}: An Extreme-Scale Linear Program Solver for Web-Applications},
  author    = {Basu, Kinjal and Ghoting, Amol and Mazumder, Rahul and Pan, Yao},
  booktitle = {Proceedings of the 37th International Conference on Machine Learning},
  pages     = {704--714},
  year      = {2020},
  volume    = {119},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  url       = {http://proceedings.mlr.press/v119/basu20a.html}
}
```

## Contributing

Please report bugs and feature requests via the [GitHub issue tracker](https://github.com/linkedin/DuaLip/issues). To contribute code, see the [contributing guide](https://linkedin.github.io/DuaLip/contributing/index.html).

## Acknowledgements

DuaLip is built and maintained by the AI Algorithms team at LinkedIn:

- [Gregory Dexter](https://www.linkedin.com/in/gregorydexter1/)
- [Aida Rahmattalabi](https://www.linkedin.com/in/aida-rahmattalabi-23a4ab4a/)
- [Sanjana Garg](https://www.linkedin.com/in/sanjana-garg/)
- [Qingquan Song](https://www.linkedin.com/in/qingquan-song-b71167119/)
- [Zhipeng Wang](https://www.linkedin.com/in/zhipeng-wang-phd-66806816/)

## License

BSD 2-Clause License. See [LICENSE](LICENSE) for details.
