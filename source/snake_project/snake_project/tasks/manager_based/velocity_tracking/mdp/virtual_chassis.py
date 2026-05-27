from __future__ import annotations

import torch

from isaaclab.utils.math import quat_from_matrix


def compute_virtual_chassis_frame(
    body_pos_w: torch.Tensor,
    prev_axes_w: torch.Tensor | None = None,
    has_prev: torch.Tensor | None = None,
    reg: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the geometric center and principal axes of the virtual chassis."""

    origin_w = body_pos_w.mean(dim=1)
    centered_body_pos_w = body_pos_w - origin_w.unsqueeze(1)
    data_matrix = centered_body_pos_w.transpose(1, 2)

    gram = torch.bmm(data_matrix, data_matrix.transpose(1, 2))
    eye = torch.eye(3, device=body_pos_w.device).unsqueeze(0)
    gram_reg = gram + reg * eye * torch.amax(gram, dim=(1, 2), keepdim=True)
    _, eigvecs = torch.linalg.eigh(gram_reg)
    axes_w = torch.flip(eigvecs, dims=[2])
    axes_w = axes_w.clone()

    if prev_axes_w is not None and has_prev is not None:
        prev_mask = has_prev.to(dtype=torch.bool)
    else:
        prev_mask = torch.zeros(body_pos_w.shape[0], dtype=torch.bool, device=body_pos_w.device)

    if prev_mask.any():
        dots = torch.sum(axes_w[prev_mask] * prev_axes_w[prev_mask], dim=1)
        signs = torch.where(dots >= 0.0, 1.0, -1.0)
        axes_w[prev_mask] = axes_w[prev_mask] * signs.unsqueeze(1)

    init_mask = ~prev_mask
    if init_mask.any():
        head_to_tail_w = body_pos_w[init_mask, -1] - body_pos_w[init_mask, 0]
        x_dots = torch.sum(axes_w[init_mask, :, 0] * head_to_tail_w, dim=1)
        x_flip = torch.where(x_dots >= 0.0, 1.0, -1.0)
        axes_w[init_mask, :, 0] = axes_w[init_mask, :, 0] * x_flip.unsqueeze(1)

        world_up = torch.zeros(int(init_mask.sum().item()), 3, device=body_pos_w.device)
        world_up[:, 2] = 1.0
        z_dots = torch.sum(axes_w[init_mask, :, 2] * world_up, dim=1)
        z_flip = torch.where(z_dots >= 0.0, 1.0, -1.0)
        axes_w[init_mask, :, 2] = axes_w[init_mask, :, 2] * z_flip.unsqueeze(1)

    det_mask = torch.det(axes_w) < 0.0
    if det_mask.any():
        axes_w[det_mask, :, 1] = -axes_w[det_mask, :, 1]

    return origin_w, axes_w


def project_world_vector_to_virtual_frame(vector_w: torch.Tensor, axes_w: torch.Tensor) -> torch.Tensor:
    """Project a world-frame vector into the virtual chassis frame."""

    return torch.einsum("bij,bj->bi", axes_w.transpose(1, 2), vector_w)


def compute_virtual_chassis_command_terms(
    body_pos_w: torch.Tensor,
    body_lin_vel_w: torch.Tensor,
    body_ang_vel_w: torch.Tensor,
    prev_axes_w: torch.Tensor | None = None,
    has_prev: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute virtual chassis frame and mean linear/angular velocity in that frame."""

    origin_w, axes_w = compute_virtual_chassis_frame(body_pos_w, prev_axes_w, has_prev)
    vc_lin_vel_w = body_lin_vel_w.mean(dim=1)
    vc_ang_vel_w = body_ang_vel_w.mean(dim=1)
    actual_lin_vel_vc = project_world_vector_to_virtual_frame(vc_lin_vel_w, axes_w)
    actual_ang_vel_vc = project_world_vector_to_virtual_frame(vc_ang_vel_w, axes_w)
    return origin_w, axes_w, actual_lin_vel_vc, actual_ang_vel_vc[:, 2]


def quat_from_axes_w(axes_w: torch.Tensor) -> torch.Tensor:
    """Convert batched orthonormal axes matrices to quaternions in (w, x, y, z)."""

    return quat_from_matrix(axes_w)


def arrow_quat_from_virtual_velocity(axes_w: torch.Tensor, velocity_vc: torch.Tensor) -> torch.Tensor:
    """Build world-frame arrow quaternions for planar velocities expressed in the virtual chassis frame."""

    yaw = torch.atan2(velocity_vc[:, 1], velocity_vc[:, 0])
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    local_rot = torch.zeros(velocity_vc.shape[0], 3, 3, device=velocity_vc.device)
    local_rot[:, 0, 0] = cos_yaw
    local_rot[:, 0, 1] = -sin_yaw
    local_rot[:, 1, 0] = sin_yaw
    local_rot[:, 1, 1] = cos_yaw
    local_rot[:, 2, 2] = 1.0
    arrow_axes_w = torch.bmm(axes_w, local_rot)
    return quat_from_axes_w(arrow_axes_w)
