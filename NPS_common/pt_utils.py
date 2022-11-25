import numpy as np
import torch


def unique_indices(arr, Nmax):
    return torch.nonzero(torch.zeros(Nmax, dtype=arr.dtype, device=arr.device).scatter_(0, arr, torch.ones_like(arr))).flatten()


def intersect_indices(arr, arr2, Nmax):
    flag = torch.zeros(Nmax, dtype=arr.dtype, device=arr.device)
    flag.scatter_(0, arr, torch.ones_like(arr))
    flag.scatter_(0, arr2, torch.ones_like(arr2))
    return torch.nonzero(flag==2).flatten()


def union_indices(arr, arr2, Nmax):
    flag = torch.zeros(Nmax, dtype=arr.dtype, device=arr.device)
    flag.scatter_(0, arr, torch.ones_like(arr))
    flag.scatter_(0, arr2, torch.ones_like(arr2))
    return torch.nonzero(flag>0).flatten()


# from https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


def broadcast_to_batch(x: torch.Tensor, ptr):
    """
    Broad cast per_graph features of shape (batch, nfeat) to batch indicated by ptr
    """
    if x.size(0) != ptr.size(0)-1:
        return x
    return x.repeat_interleave(ptr[1:]-ptr[:-1], dim=0)

