import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch

from dualip.objectives.matching import MatchingInputArgs
from dualip.projections.base import create_projection_map
from dualip.run_solver import run_solver
from dualip.types import ComputeArgs, ObjectiveArgs, SolverArgs


@dataclass
class MovielensMatchingConfig:
    ratings_csv_path: str
    per_movie_capacity: float = 1.0
    # Rating transform -> reward; c_ij = -reward_ij
    rating_scale: float = 1.0
    rating_shift: float = 0.0
    # Optional filtering
    min_user_interactions: int = 1
    min_movie_interactions: int = 1
    # Device for result tensors
    device: str = "cpu"


def _build_index_maps(user_ids: Iterable[int], movie_ids: Iterable[int]) -> Tuple[Dict[int, int], Dict[int, int]]:
    unique_users = np.unique(np.fromiter(user_ids, dtype=np.int64))
    unique_movies = np.unique(np.fromiter(movie_ids, dtype=np.int64))
    user_id_to_col = {int(u): i for i, u in enumerate(unique_users)}
    movie_id_to_row = {int(m): j for j, m in enumerate(unique_movies)}
    return user_id_to_col, movie_id_to_row


def _compute_budgets(
    movie_id_to_row: Dict[int, int],
    per_movie_capacity: float,
    device: str,
) -> torch.Tensor:
    num_rows = len(movie_id_to_row)
    b_vals = np.zeros(num_rows, dtype=np.float32)
    b_vals[:] = float(per_movie_capacity)
    return torch.tensor(b_vals, dtype=torch.float32, device=device)


def _ratings_to_c_and_a(
    df: pd.DataFrame,
    user_id_to_col: Dict[int, int],
    movie_id_to_row: Dict[int, int],
    rating_scale: float,
    rating_shift: float,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build CSC matrices C (objective coefficients) and A (costs) of shape (num_movies, num_users).
    We use a_ij = 1 for all observed pairs and c_ij = -(scale * rating + shift).
    Torch CSC requires per-column row indices to be sorted and distinct.
    """
    num_cols = len(user_id_to_col)
    num_rows = len(movie_id_to_row)

    # Group entries by user (column); deduplicate (row_idx) within each column
    # by keeping the best reward (i.e., min c since c = -reward).
    per_user_entries: Dict[int, Dict[int, Tuple[float, float]]] = defaultdict(dict)
    for _, row in df.iterrows():
        u = int(row["userId"])
        m = int(row["movieId"])
        r = float(row["rating"])
        col = user_id_to_col[u]
        row_idx = movie_id_to_row[m]
        reward = rating_scale * r + rating_shift
        cval = -reward
        aval = 1.0
        existing = per_user_entries[col].get(row_idx)
        if existing is None:
            per_user_entries[col][row_idx] = (cval, aval)
        else:
            # Keep the more favorable (smaller) c value for duplicates
            if cval < existing[0]:
                per_user_entries[col][row_idx] = (cval, aval)

    # Build CSC buffers
    col_ptr = [0]
    row_idx: List[int] = []
    vals_C: List[float] = []
    vals_A: List[float] = []
    nnz = 0
    for col in range(num_cols):
        # Sort strictly by row index ascending to satisfy CSC invariant
        col_entries = per_user_entries.get(col, {})
        if not col_entries:
            # Empty column; leave as is (col_ptr unchanged) to represent zero nnz
            col_ptr.append(nnz)
            continue
        for ridx in sorted(col_entries.keys()):
            cval, aval = col_entries[ridx]
            row_idx.append(ridx)
            vals_C.append(cval)
            vals_A.append(aval)
            nnz += 1
        col_ptr.append(nnz)

    col_ptr_t = torch.tensor(col_ptr, dtype=torch.int64, device=device)
    row_idx_t = torch.tensor(row_idx, dtype=torch.int64, device=device)
    vals_C_t = torch.tensor(vals_C, dtype=torch.float32, device=device)
    vals_A_t = torch.tensor(vals_A, dtype=torch.float32, device=device)

    C = torch.sparse_csc_tensor(col_ptr_t, row_idx_t, vals_C_t, size=(num_rows, num_cols), device=device)
    A = torch.sparse_csc_tensor(col_ptr_t, row_idx_t, vals_A_t, size=(num_rows, num_cols), device=device)
    return C, A


def prepare_movielens_matching(
    config: MovielensMatchingConfig,
) -> Tuple[MatchingInputArgs, Dict[str, int], Dict[int, int]]:
    """
    Convert MovieLens ratings to MatchingInputArgs consumable by dualip_matching.
    Columns (users) i, rows (movies) j.
    - A[j, i] = 1 for observed (user i, movie j)
    - c[j, i] = -(scale * rating + shift)
    - projection_map: per-user simplex or simplex_eq with z = 1
    - b_vec[j] = budget

    Returns:
        (input_args, user_id_to_col, row_to_movie_id)
    """
    df = pd.read_csv(config.ratings_csv_path, usecols=["userId", "movieId", "rating"])

    # Optional filtering to drop sparse users/movies
    if config.min_user_interactions > 1:
        user_counts = df["userId"].value_counts()
        keep_users = set(user_counts[user_counts >= config.min_user_interactions].index.astype(int).tolist())
        df = df[df["userId"].isin(keep_users)]
    if config.min_movie_interactions > 1:
        movie_counts = df["movieId"].value_counts()
        keep_movies = set(movie_counts[movie_counts >= config.min_movie_interactions].index.astype(int).tolist())
        df = df[df["movieId"].isin(keep_movies)]

    user_id_to_col, movie_id_to_row = _build_index_maps(df["userId"].astype(int), df["movieId"].astype(int))
    print(f"Number of users: {len(user_id_to_col)}, number of movies: {len(movie_id_to_row)}")
    # Build C and A
    C, A = _ratings_to_c_and_a(
        df,
        user_id_to_col=user_id_to_col,
        movie_id_to_row=movie_id_to_row,
        rating_scale=config.rating_scale,
        rating_shift=config.rating_shift,
        device=config.device,
    )
    print(f"Built matching inputs: A shape {tuple(A.shape)}, C shape {tuple(C.shape)}")
    # Budgets
    b_vec = _compute_budgets(
        movie_id_to_row=movie_id_to_row,
        per_movie_capacity=config.per_movie_capacity,
        device=config.device,
    )
    print(f"b_vec shape: {tuple(b_vec.shape)}")
    projection_map = create_projection_map("simplex", {"z": 1}, num_indices=len(user_id_to_col))

    input_args = MatchingInputArgs(A=A, c=C, projection_map=projection_map, b_vec=b_vec, equality_mask=None)

    # Also return a reverse map for convenience when inspecting rows
    row_to_movie_id = {row: movie for movie, row in movie_id_to_row.items()}
    return input_args, user_id_to_col, row_to_movie_id


def _save_snapshot(
    input_args: MatchingInputArgs,
    out_prefix: str,
    user_id_to_col: Dict[int, int],
    row_to_movie_id: Dict[int, int],
) -> None:
    # Always save snapshots on CPU to avoid CUDA deserialization issues with sparse CSC tensors.
    torch.save(
        {
            "A": input_args.A.to("cpu"),
            "c": input_args.c.to("cpu"),
            "b_vec": input_args.b_vec.to("cpu"),
        },
        f"{out_prefix}.pt",
    )
    with open(f"{out_prefix}_user_map.json", "w") as f:
        json.dump(user_id_to_col, f)
    with open(f"{out_prefix}_row_to_movie.json", "w") as f:
        json.dump(row_to_movie_id, f)


def _load_snapshot(
    in_prefix: str,
) -> Tuple[MatchingInputArgs, Dict[str, int], Dict[int, int]]:
    """
    Load a previously saved snapshot (CPU tensors) and reconstruct MatchingInputArgs.
    Note: tensors are kept on CPU; downstream code moves them to the compute device.
    """
    payload = torch.load(f"{in_prefix}.pt", map_location="cpu")
    A = payload["A"].to("cpu")
    c = payload["c"].to("cpu")
    b_vec = payload["b_vec"].to("cpu")
    # Rebuild projection map (simplex with z=1) based on number of users/columns
    num_cols = A.size(1)
    projection_map = create_projection_map("simplex", {"z": 1}, num_indices=num_cols)
    input_args = MatchingInputArgs(A=A, c=c, projection_map=projection_map, b_vec=b_vec, equality_mask=None)

    with open(f"{in_prefix}_user_map.json", "r") as f:
        user_id_to_col = json.load(f)
        # keys may be strings after JSON round-trip; normalize to int
        user_id_to_col = {int(k): int(v) for k, v in user_id_to_col.items()}
    with open(f"{in_prefix}_row_to_movie.json", "r") as f:
        row_to_movie_id = json.load(f)
        row_to_movie_id = {int(k): int(v) for k, v in row_to_movie_id.items()}

    return input_args, user_id_to_col, row_to_movie_id


def main():
    parser = argparse.ArgumentParser(description="Build matching inputs from MovieLens ratings for dualip_matching.")
    parser.add_argument("--ratings_csv_path", type=str, required=True)
    parser.add_argument("--per_movie_capacity", type=float, default=30.0)
    parser.add_argument("--rating_scale", type=float, default=1.0)
    parser.add_argument("--rating_shift", type=float, default=0.0)
    parser.add_argument("--min_user_interactions", type=int, default=1)
    parser.add_argument("--min_movie_interactions", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run_solver", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--max_iter", type=int, default=10000)
    parser.add_argument("--initial_step_size", type=float, default=0.00000001)
    parser.add_argument("--max_step_size", type=float, default=0.000001)
    parser.add_argument("--out_prefix", type=str, default=None, help="If set, save a snapshot of A/c/b and mappings")
    parser.add_argument(
        "--in_prefix",
        type=str,
        default=None,
        help="If set, load a prebuilt snapshot of A/c/b and mappings instead of rebuilding from CSV",
    )
    args = parser.parse_args()

    if args.in_prefix:
        print(f"Loading snapshot from prefix: {args.in_prefix}")
        input_args, user_map, row_to_movie = _load_snapshot(args.in_prefix)
    else:
        config = MovielensMatchingConfig(
            ratings_csv_path=args.ratings_csv_path,
            per_movie_capacity=args.per_movie_capacity,
            rating_scale=args.rating_scale,
            rating_shift=args.rating_shift,
            min_user_interactions=args.min_user_interactions,
            min_movie_interactions=args.min_movie_interactions,
            device=args.device,
        )
        input_args, user_map, row_to_movie = prepare_movielens_matching(config)

    if args.out_prefix:
        _save_snapshot(input_args, args.out_prefix, user_map, row_to_movie)

    if args.run_solver:
        host_device = args.device
        solver_args = SolverArgs(
            gamma=args.gamma,
            max_iter=args.max_iter,
            initial_step_size=args.initial_step_size,
            max_step_size=args.max_step_size,
        )
        compute_args = ComputeArgs(compute_device_num=1, host_device=host_device)
        objective_args = ObjectiveArgs(objective_type="matching")
        result = run_solver(
            input_args=input_args,
            solver_args=solver_args,
            compute_args=compute_args,
            objective_args=objective_args,
        )
        print("Dual objective:", result.dual_objective)
        print(
            "A shape:",
            tuple(input_args.A.shape),
            "C shape:",
            tuple(input_args.c.shape),
            "b shape:",
            tuple(input_args.b_vec.shape),
        )
    else:
        print("Prepared matching inputs.")
        print(
            "A shape:",
            tuple(input_args.A.shape),
            "C shape:",
            tuple(input_args.c.shape),
            "b shape:",
            tuple(input_args.b_vec.shape),
        )


if __name__ == "__main__":
    main()
