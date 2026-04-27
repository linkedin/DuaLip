from collections import defaultdict
from dataclasses import fields

import torch

from dualip.objectives.base import BaseInputArgs
from dualip.projections.base import ProjectionEntry
from dualip.utils.sparse_utils import split_csc_by_cols


def transfer_tensors_to_device(input_args: BaseInputArgs, device: str):
    """
    Transfer all tensor fields in input_args to the specified device.

    Args:
        input_args: The input arguments dataclass.
        device: The target device (e.g., 'cuda:0', 'cpu').

    Returns:
        A new instance of the same dataclass with all tensors transferred to device.
    """
    field_values = {}
    for field in fields(input_args):
        value = getattr(input_args, field.name)
        if isinstance(value, torch.Tensor):
            field_values[field.name] = value.to(device)
        else:
            field_values[field.name] = value
    return type(input_args)(**field_values)


def global_to_local_projection_map(global_map: dict[str, ProjectionEntry], local_cols: list[int]) -> dict[str, dict]:
    """
    Given a global projection_map and the list of global col indices for a split,
    return a local projection_map with local indices. Example:
    """
    global2local = {g: loc for loc, g in enumerate(local_cols)}  # map from global idx to local idx
    local_map = defaultdict(ProjectionEntry)

    for proj_key, proj_item in global_map.items():
        indices = proj_item.indices
        local_indices = [global2local[g] for g in indices if g in global2local]
        if local_indices:
            pe = local_map[proj_key]
            pe.indices = local_indices
            pe.proj_params = proj_item.proj_params
            pe.proj_type = proj_item.proj_type
    return local_map


def split_tensors_to_devices(a_mat: torch.Tensor, c_mat: torch.Tensor, compute_devices: list) -> tuple:
    """
    Split the CSC-format input tensors across multiple devices in a balanced way.

    Args:
        a_mat: The compact A matrix tensor of shape (m, n)
        c_mat: The 2D C matrix tensor of shape (m, n)
        compute_devices: List of device strings (e.g., ['cuda:0', 'cuda:1'])

    Returns:
        tuple: (split_a_tensors, split_c_tensors) where each is a list of tensors on different devices
    """

    if a_mat.layout != torch.sparse_csc or c_mat.layout != torch.sparse_csc:
        raise ValueError("Both A and B must be CSC-format sparse tensors")

    if not compute_devices:
        num_cols = a_mat.size(1)
        split_index_map = [i for i in range(num_cols)]
        return [a_mat], [c_mat], split_index_map

    num_devices = len(compute_devices)
    num_cols = a_mat.size(1)

    # Calculate how many blocks per device, ensuring we keep blocks together
    base_cols_per_device = num_cols // num_devices
    remainder_blocks = num_cols % num_devices

    # Calculate split sizes in terms of columns (block_size * number of blocks)
    split_sizes = [(base_cols_per_device + (1 if i < remainder_blocks else 0)) for i in range(num_devices)]
    split_index_map = []
    start = 0
    for _, size in enumerate(split_sizes):
        split_index_map.append(list(range(start, start + size)))  # global indices
        start += size

    # Split tensors along column dimension (dim=1) for A and along first dimension for c
    a_splits = split_csc_by_cols(a_mat, split_sizes)
    c_splits = split_csc_by_cols(c_mat, split_sizes)

    # Move splits to respective devices
    a_device_splits = [split.to(device) for split, device in zip(a_splits, compute_devices)]
    c_device_splits = [split.to(device) for split, device in zip(c_splits, compute_devices)]
    return a_device_splits, c_device_splits, split_index_map
